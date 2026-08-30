import subprocess
import time
import torch
import numpy as np
import re
import os
import signal
import math

from model import query_gpt
from model.embedding import OpenAIEmbedding
from utils.file_utils import read_code, read_text

FLOAT_TOLERANCE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

class BenchmarkEvaluator:
    @staticmethod
    def commongen_coverage(concepts, text_path):
        generated_text = read_text(text_path)
        generated_text = generated_text.lower()
        concepts = [concept.lower() for concept in concepts]
        missing_concepts = [concept for concept in concepts if not re.search(rf'\b{re.escape(concept)}\b', generated_text, re.IGNORECASE)]
        if missing_concepts:
            return 1 - len(missing_concepts)/len(concepts)
        return 1

    @staticmethod
    def commongen_gpt_score(concepts, text_path):
        generated_text = read_text(text_path)
        prompt = '''
        As a strict StoryMaster, your task is to meticulously evaluate the quality of stories across three primary dimensions: Grammar and Fluency, Context Relevance, and Logic Consistency. Each dimension will be rated on a refined scale from 1 (average) to 4 (perfect), ensuring that only stories of superior quality achieve the highest scores.

        Implement Your Evaluation Mechanism with Enhanced Rigor:

        Grammar and Fluency (Assess the story's linguistic precision and narrative flow):
        Score 1 (solid): The story is free of grammatical errors, but the narrative lacks the stylistic variety and eloquence that elevate writing to a higher tier.
        Score 2 (proficient): The narrative demonstrates a strong command of grammar and a coherent flow, yet it does not showcase the level of linguistic artistry found in superior works.
        Score 3 (excellent): The story exhibits a refined sense of grammar and a compelling narrative flow, with sentence structures that are engaging and demonstrate a high level of craft.
        Score 4 (masterful): The story is a testament to linguistic excellence, with sentence structures that are not only clear and elegant but also exhibit a creative and sophisticated use of language that captivates and inspires.

        Context Relevance (Examine the coherence, interconnectedness, and depth of content within the story):
        Score 1 (solid): The story establishes a basic framework of context relevance, but it does not delve into the intricacies of character and thematic development that enrich the narrative.
        Score 2 (proficient): The narrative demonstrates a clear connection between elements, yet it lacks the depth and multi-layered content that would distinguish it as truly exceptional.
        Score 3 (excellent): The story interweaves elements with a high degree of relevance, creating a narrative that is coherent and features content that is well-developed and insightful.
        Score 4 (masterful): The story achieves an extraordinary level of context relevance, with every element artfully woven into a narrative that is not only coherent but also profound in its exploration of themes and characters, offering a rich and immersive experience.

        Logic Consistency (Scrutinize the narrative for logical integrity and internal consistency):
        Score 1 (solid): The story maintains a logical structure, but there may be occasional lapses in plausibility or minor inconsistencies that slightly undermine its credibility.
        Score 2 (proficient): The narrative is generally logical, with a clear progression of events and character actions, yet it does not reach the level of seamless consistency expected of a superior story.
        Score 3 (excellent): The story exhibits a strong logical consistency, with events and character actions that are well-aligned and plausible, contributing to a coherent and believable plot.
        Score 4 (masterful): The story is characterized by impeccable logical consistency, with every event and character action meticulously aligned to create a plot that is not only coherent but also demonstrates a deep understanding of causality and human behavior.'''

        prompt += '\nStory:\n' + generated_text
        response_text, _ = query_gpt(prompt)
        pattern = r'\d+'
        remedy_prompt = 'Extract the score in each dimension in format: (Grammar and Fluency Score: X. Context Relevance Score: X. Logic Consistency Score: X. Overall Score Score: X.) of the following content.'
        remedy_prompt += response_text
        remedy_respond,_ = query_gpt(remedy_prompt)
        score_list = re.findall(pattern, remedy_respond)
        my_float_list = [float(item) for item in score_list]
        score_list = [item/4 for item in my_float_list]
        score_list = score_list[:3]
        while len(score_list) != 3:
            score_list.append(0)
        return score_list

    @staticmethod
    def check_commongen(concepts, text_path):
        # Metric implementation inspired by self-refine project:
        # https://github.com/madaan/self-refine/tree/main/src/commongen
        coverage = BenchmarkEvaluator.commongen_coverage(concepts, text_path)
        coverage = torch.tensor(coverage, dtype=torch.float32, device=DEVICE)  
        scores = BenchmarkEvaluator.commongen_gpt_score(concepts, text_path)
        grammar = torch.tensor(scores[0], dtype=torch.float32, device=DEVICE)  
        relevance = torch.tensor(scores[1], dtype=torch.float32, device=DEVICE)  
        consistency = torch.tensor(scores[2], dtype=torch.float32, device=DEVICE)  
        metrics = {"grammar": grammar, "relevance": relevance, "consistency": consistency, "coverage": coverage}
        mean_score = torch.tensor(sum(scores) / 3, dtype=torch.float32, device=DEVICE)    
        if coverage == 0:
            return -1.0, metrics
        else:
            return coverage*mean_score, metrics
        
    
    @staticmethod
    def check_srdd(code_path, text):
        # Metric implementation inspired by ChatDev project:
        # https://github.com/OpenBMB/ChatDev
        path = code_path
        code = read_code(path)
        consistency = BenchmarkEvaluator.srdd_consistency(text, code)
        completeness = BenchmarkEvaluator.srdd_completeness(code)
        executability, _ = BenchmarkEvaluator.srdd_executability(path)
        executability = 1 if executability else 0
        executability = torch.tensor(executability, dtype=torch.float32, device=DEVICE)  
        consistency = torch.tensor(consistency, dtype=torch.float32, device=DEVICE)  
        completeness = torch.tensor(completeness, dtype=torch.float32, device=DEVICE)  
        metrics = {"consistency": consistency, "completeness": completeness, "executability": executability}
        reward = (executability + consistency + completeness) / 3
        return reward, metrics
    
    @staticmethod
    def srdd_consistency(text, code):
        code = BenchmarkEvaluator.remove_comments(code)
        text = re.sub(r'^[^\n]*\n', '', text)
        text_embedding = OpenAIEmbedding.get_embedding(text)
        code_embedding = OpenAIEmbedding.get_embedding(code)
        similarity = BenchmarkEvaluator.get_cosine_similarity(text_embedding, code_embedding)
        return similarity

    @staticmethod
    def srdd_completeness(code):
        lines = code.split("\n")
        lines = [line for line in lines if
                "password" not in line.lower() and "passenger" not in line.lower() and "passed" not in line.lower() and "passes" not in line.lower()]
        lines = [line for line in lines if "pass" in line.lower() or "todo" in line.lower()]
        if len(lines) > 0:
            return 0.0
        return 1.0

    @staticmethod 
    def srdd_executability(work_path):
        def robust_kill(process):
            """Robustly kill the process based on the OS."""
            if process.poll() is None:  # Check if the process is still running
                if os.name == 'nt':  # For Windows
                    os.kill(process.pid, signal.SIGTERM)
                    time.sleep(1)  
                    if process.poll() is None:  
                        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:  # For Linux/macOS
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  
                    time.sleep(1)  
                    if process.poll() is None:  
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        try:
            if not os.path.exists(work_path):
                return False, "The file path does not exist."
            
            # Use mas conda environment Python interpreter
            python_executable = "/root/miniconda3/envs/mas/bin/python"
            
            if os.name == 'nt':  
                command = f"{python_executable} {work_path}"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  
                command = f"{python_executable} {work_path}"
                process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

            try:
                out, err = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                robust_kill(process)
                return True, "The process completes without encountering any errors."

            return_code = process.returncode
            output = out.decode('utf-8', errors='ignore')
            error_output = err.decode('utf-8', errors='ignore')

            # If the process is still running after the timeout
            if process.poll() is None:
                robust_kill(process)  
            return_code = process.returncode

            # Handle return code and output
            if return_code == 0:
                # Clean up file paths in the output for readability
                work_path = os.getcwd()
                output = output.replace(work_path, "")
                return True, output
            else:
                # Handle errors in the output
                if error_output:
                    work_path = os.getcwd()
                    if "Traceback".lower() in error_output.lower():
                        errs = error_output.replace(work_path + "/", "").replace(work_path, "")
                        return False, errs
                return False, error_output

        except subprocess.CalledProcessError as e:
            return False, f"CalledProcessError: {str(e)}"
        except Exception as ex:
            return False, f"An unexpected error occurred: {str(ex)}"


    @staticmethod
    def get_cosine_similarity(embeddingi, embeddingj):
        embeddingi = np.array(embeddingi)
        embeddingj = np.array(embeddingj).T
        cos_sim = embeddingi.dot(embeddingj) / (np.linalg.norm(embeddingi) * np.linalg.norm(embeddingj))
        return cos_sim
    
    @staticmethod
    def remove_comments(string):
        def remove_comments_by_regex(string, regex):
            lines = string.split("\n")
            lines = [line for line in lines if not line.strip().startswith("#")]
            string = "\n".join(lines)
            comments = []
            matches = re.finditer(regex, string, re.DOTALL)
            for match in matches:
                group1 = match.group(1)
                comments.append(group1)
            for comment in comments + ["''''''\n"]:
                string = string.replace(comment, "")
            return string

        string = remove_comments_by_regex(string, r"'''(.*?)'''")
        string = remove_comments_by_regex(string, r"\"\"\"(.*?)\"\"\"")
        return string

    
    @staticmethod
    def check_mmlu(final_ans, true_ans, options=None):
        """
        MMLU-Pro 判分逻辑。
        
        使用增强版 extract_mmlu_answer 从混乱的模型输出中提取单个选项字母。
        
        Args:
            final_ans: 模型输出的答案
            true_ans: 正确答案（字母）
            options: 选项列表，用于支持内容匹配
        """
        if final_ans is None or true_ans is None:
            return False
        if len(str(final_ans)) == 0:
            return False
        
        # 使用增强版答案提取，传入options支持内容匹配
        pred = BenchmarkEvaluator.extract_mmlu_answer(final_ans, options=options)
        gold = BenchmarkEvaluator.extract_mmlu_answer(true_ans)
        
        if pred is None or gold is None:
            return False
        
        return pred.upper() == gold.upper()
    
    @staticmethod
    def check_gsm8k(final_ans, true_ans):
        if final_ans is None or true_ans is None:   
            return False
        if isinstance(final_ans, str):
            final_num = BenchmarkEvaluator.extract_number(final_ans)
            if final_num is None:
                return False
        else:
            final_num = float(final_ans)
        true_num = float(true_ans)
        
        if not (math.isfinite(final_num) and math.isfinite(true_num)):
            return False  

        # Accuracy computation adapted from: https://github.com/reasoning-machines/pal/blob/main/scripts/gsm_eval.py
        is_correct = abs(float(final_num) - float(true_num)) < FLOAT_TOLERANCE 
        if not is_correct:
            is_correct = (round(float(final_num)) == round(float(true_num)))
            if is_correct:
                 return is_correct
            if abs(int(float(final_num))) > 100 and abs(int(float(true_num))) > 100:
                is_correct = (int(float(final_num)) == int(float(true_num)))
        return is_correct
    
    @staticmethod
    def extract_math_answer(text):
        if text is None:
            return text
        if isinstance(text, str):
            final_num = BenchmarkEvaluator.extract_number(text)
        else:
            final_num = float(text)
        return final_num
    
    @staticmethod
    def extract_choice_answer(text):
        """从文本中提取单个选项字母 (A-J)，用于 MMLU-Pro 等多选题。"""
        return BenchmarkEvaluator.extract_mmlu_answer(text)
    
    @staticmethod
    def extract_mmlu_answer(text, options=None):
        """
        增强版 MMLU-Pro 答案提取。
        
        从可能很乱的模型输出中提取单个选项字母 (A-J)。
        支持多种格式：
        - "FINAL ANSWER: A"
        - "The answer is A"
        - "answer is (A)"
        - "**A**"
        - 纯字母 "A"
        - 混杂在推理链中的答案
        - **NEW**: 如果输出内容匹配某个选项的内容，提取对应字母
        
        Args:
            text: 模型输出文本
            options: 选项列表，格式如 ["A: $8.35", "B: $10.50", ...]
        
        优先级：
        1. FINAL ANSWER 后的字母
        2. answer is 后的字母
        3. boxed 中的字母
        4. 括号中的字母 (A)
        5. 冒号后的字母 : A
        6. 加粗的字母 **A**
        7. 最后出现的单独字母
        8. 第一个出现的单独字母
        9. **NEW**: 匹配选项内容
        """
        if text is None:
            return None
        
        text = str(text)
        
        # 清理转义字符
        text = text.replace('\\n', '\n').replace("\\'", "'")
        
        # 有效选项字母
        valid_letters = set('ABCDEFGHIJ')
        
        # Pattern 1: FINAL ANSWER: X 或 FINAL ANSWER: (X)
        patterns_priority = [
            r'FINAL\s*ANSWER[:\s]+\(?([A-J])\)?',
            r'final\s*answer[:\s]+\(?([A-Ja-j])\)?',
            r'[Tt]he\s+answer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
            r'[Aa]nswer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
            r'[Cc]orrect\s+answer\s+is\s*[:\s]?\s*\(?([A-Ja-j])\)?',
            r'[Bb]oxed\{([A-Ja-j])\}',
            r'\\boxed\{([A-Ja-j])\}',
        ]
        
        for pattern in patterns_priority:
            match = re.search(pattern, text)
            if match:
                letter = match.group(1).upper()
                if letter in valid_letters:
                    return letter
        
        # Pattern 2: 括号中的字母，如 (A) 或 [A]
        paren_patterns = [
            r'\(([A-J])\)',
            r'\[([A-J])\]',
        ]
        for pattern in paren_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # 取最后一个（通常是最终答案）
                letter = matches[-1].upper()
                if letter in valid_letters:
                    return letter
        
        # Pattern 3: 冒号后的字母，如 "Answer: A" 或 ": A"
        # 但要确保冒号前有答案相关的关键词，避免误提取
        colon_pattern = r'(answer|choice|option|select|choose|correct|pick|is|result)[\s:]+([A-J])(?:[\s,.:;\)]|$)'
        matches = re.findall(colon_pattern, text, re.IGNORECASE)
        if matches:
            letter = matches[-1][1].upper()  # matches是元组列表，取第二个元素
            if letter in valid_letters:
                return letter
        
        # Pattern 4: 加粗的字母 **A** 或 *A*
        bold_pattern = r'\*+([A-J])\*+'
        match = re.search(bold_pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
        
        # Pattern 5: 开头或结尾独立的字母
        # 检查开头（允许字母后跟引号、括号、星号等多种字符）
        start_pattern = r'^\s*([A-J])(?:[\s,.:;)"\'\\ *]|$)'
        match = re.search(start_pattern, text.strip(), re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
        
        # 检查结尾（取最后一行）
        # 但要排除常见的英文单词（如"I"）
        lines = text.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        end_pattern = r'(?:^|[\s,.:;\(])([A-J])\s*[.!)?]*\s*$'
        match = re.search(end_pattern, last_line, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            # 排除单独的"I"（英文单词）
            if letter == 'I' and re.search(r'\bi\s*$', last_line, re.IGNORECASE):
                pass  # 跳过
            elif letter in valid_letters:
                return letter
        
        # Pattern 6: 禁用独立单字母匹配
        # 原因：容易误提取"I think"、"a solution"等英文单词中的字母
        # 如果前面的明确模式都没匹配到，说明答案格式不规范，应该返回None触发强制回答
        
        # Pattern 7: 文本长度为1且是字母
        text_stripped = text.strip()
        if len(text_stripped) == 1 and text_stripped.upper() in valid_letters:
            return text_stripped.upper()
        
        # Pattern 8: 匹配选项内容（NEW）
        # 注意：移除了原来的Pattern 8 Fallback机制（取第一个字母），避免误提取"Based"等情况
        # 如果提供了选项列表，尝试匹配输出内容到选项内容
        if options:
            text_clean = text.strip().lower()
            # 移除常见的前缀词
            text_clean = re.sub(r'^(the answer is|answer is|answer:|final answer:)\s*', '', text_clean, flags=re.IGNORECASE)
            text_clean = text_clean.strip()
            
            # 标准化函数：移除空格、货币符号、逗号等噪声
            def normalize_value(s):
                """标准化数值/金额字符串，移除空格、货币符号、逗号等"""
                s = s.strip()
                # 移除货币符号
                s = re.sub(r'[\$€£¥]', '', s)
                # 移除逗号
                s = s.replace(',', '')
                # 移除多余空格
                s = re.sub(r'\s+', '', s)
                # 移除尾部的标点符号（如括号、引号等）
                s = re.sub(r'[)\]"\'\s]+$', '', s)
                return s.strip()
            
            for option in options:
                # 选项格式: "A: $8.35" 或 "A: Some text"
                match = re.match(r'^([A-J]):\s*(.+)$', option.strip(), re.IGNORECASE)
                if match:
                    letter = match.group(1).upper()
                    content = match.group(2).strip()
                    
                    # 检查模型输出是否包含或匹配选项内容
                    # 1. 完全匹配（原始）
                    if text_clean == content.lower():
                        return letter
                    
                    # 2. 模型输出包含选项内容
                    if content.lower() in text_clean:
                        return letter
                    
                    # 3. 对于数字/金额，标准化后比较
                    if re.search(r'[\d\$€£¥,.]+', content):
                        # 提取选项中的数字/金额
                        option_numbers = re.findall(r'[\$€£¥]?[\d,]+\.?\d*', content)
                        text_numbers = re.findall(r'[\$€£¥]?[\d,]+\.?\d*', text_clean)
                        
                        # 标准化并比较
                        for opt_num in option_numbers:
                            opt_num_normalized = normalize_value(opt_num)
                            if not opt_num_normalized:
                                continue
                            
                            for text_num in text_numbers:
                                text_num_normalized = normalize_value(text_num)
                                if not text_num_normalized:
                                    continue
                                
                                # 尝试数值比较
                                try:
                                    opt_val = float(opt_num_normalized)
                                    text_val = float(text_num_normalized)
                                    if abs(opt_val - text_val) < 0.01:
                                        return letter
                                except:
                                    # 字符串比较
                                    if opt_num_normalized == text_num_normalized:
                                        return letter
        
        return None
    
    @staticmethod
    def normalize_string(s):
        return ''.join(s.split()).lower()

    @staticmethod
    def extract_number(text):
        """
        从文本中提取数值，支持科学计数法。
        
        支持格式:
        - 普通数值: 50.7, -1368, +65.49
        - 科学计数法: 2.88e-10, 2.88E10
        - LaTeX科学计数法: 2.88 \\times 10^{-10}, 2.88 × 10^{-10}
        """
        if text is None:
            return None
        text = str(text)
        
        # 1. 先尝试匹配 LaTeX 科学计数法: a \times 10^{b} 或 a × 10^{b}
        latex_sci_pattern = r'(-?\d+\.?\d*)\s*(?:\\times|×)\s*10\^?\{?(-?\d+)\}?'
        latex_match = re.search(latex_sci_pattern, text)
        if latex_match:
            base = float(latex_match.group(1))
            exp = int(latex_match.group(2))
            return base * (10 ** exp)
        
        # 2. 尝试匹配标准科学计数法: aEb 或 aeb
        std_sci_pattern = r'-?\d+\.?\d*[eE][+-]?\d+'
        std_match = re.search(std_sci_pattern, text)
        if std_match:
            return float(std_match.group())
        
        # 3. 匹配普通数值 (带符号)
        num_pattern = r'[+-]?\d+\.\d+|[+-]?\d+'
        matches = re.findall(num_pattern, text)
        return float(matches[0]) if matches else None

    @staticmethod
    def extract_ground_truth(text):
        return text.split('####')[-1].strip()
    
    @staticmethod
    def extract_letter(text):
            pattern = r'\((\w)\)'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()  
            return text.strip()
    
    @staticmethod
    def _coerce_to_text(value) -> str:
        """
        将模型输出转换为纯文本字符串。
        
        MAS runner 可能返回:
        - 纯文本字符串
        - dict/tuple（某些工具链返回结构化信息）
        - 代码执行输出
        
        处理逻辑:
        - list/tuple 取第一个元素
        - dict 优先取 answer/final_answer/text/content 等字段
        - 最后 fallback 到 str(value)
        """
        if value is None:
            return ""
        
        # 如果是 tuple 或 list，取第一个元素
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            value = value[0]
        
        # 如果是 dict，尝试提取常见字段
        if isinstance(value, dict):
            for key in ['answer', 'final_answer', 'Answer', 'text', 'content', 'result', 'output']:
                if key in value and value[key] is not None:
                    return str(value[key])
            # Fallback: 转换整个 dict
            return str(value)
        
        return str(value)
    
    @staticmethod
    def check_scibench(final_ans, true_ans) -> bool:
        """
        SciBench 判分逻辑。
        
        评测流程:
        1. 将 gold (true_ans) 转为 float
        2. 从 pred (final_ans) 文本中抽取数值，转为 float
        3. 容差判断: tol = max(1e-3, abs(gold) * 1e-3)
        
        满足任一条件即算正确:
        - abs(pred - gold) <= tol (绝对+相对容差)
        - round(pred) == round(gold) (四舍五入相等)
        - 两者都大于 100 时 int(pred) == int(gold) (大数鲁棒)
        
        Args:
            final_ans: 模型预测输出
            true_ans: 标准答案 (answer_number 字符串)
        
        Returns:
            是否正确
        """
        if final_ans is None or true_ans is None:
            return False
        
        # 1. 将 gold 转为 float
        try:
            # true_ans 可能是 "+65.49" 或 "0" 等字符串
            gold = float(str(true_ans).strip())
        except (ValueError, TypeError):
            return False
        
        # 2. 从预测文本中抽取数值
        pred_text = BenchmarkEvaluator._coerce_to_text(final_ans)
        pred_num = BenchmarkEvaluator.extract_number(pred_text)
        
        if pred_num is None:
            return False
        
        # 检查是否为有效数值
        if not (math.isfinite(pred_num) and math.isfinite(gold)):
            return False
        
        # 3. 容差判断
        # tol = max(1e-3, abs(gold) * 1e-3) - 绝对 + 相对容差
        tol = max(1e-3, abs(gold) * 1e-3)
        
        # 条件1: 绝对+相对容差
        if abs(pred_num - gold) <= tol:
            return True
        
        # 条件2: 四舍五入相等
        if round(pred_num) == round(gold):
            return True
        
        # 条件3: 大数鲁棒 - 两者都大于 100 时比较整数部分
        if abs(pred_num) > 100 and abs(gold) > 100:
            if int(pred_num) == int(gold):
                return True
        
        return False  