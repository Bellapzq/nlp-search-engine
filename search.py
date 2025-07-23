import re
import string
import nltk
import sys
import os
import difflib
import itertools

nltk.download('averaged_perceptron_tagger', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)
nltk.download('punkt', quiet = True)
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from itertools import product

def load_index(folder_of_indexes):
    index = {}
    index_file_path = os.path.join(folder_of_indexes, 'index.txt')
    with open(index_file_path, 'r', encoding='utf-8') as index_file:
        for line in index_file:
            # 假设文件格式为：word: {doc_id: {position: lineNumber, ...}}
            word, doc_info = line.split(":", 1)
            word = word.strip()
            index[word] = eval(doc_info.strip())  # 将字符串解析成字典
    return index

# 句子语法处理
# 初始化 lemmatizer 和 stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def process_sentence_grammar_2(sentence):
    # 定义名词后缀列表
    sentence = sentence.lower()
    noun_suffixes = ["tion", "tions", "ings", "yure", "ent", "our", "cian"]

    sentence = re.sub(r"\b(\w+)'s\b", r"\1s", sentence)  # 处理 "cat's"
    sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence)  # 处理 "cats'"
    sentence = sentence.replace('-', ' ')
    sentence = sentence.replace('.', '')
    
    sentence = re.sub(r"\b(can|won|is|are|do|does|did|has|have|had|could|would|should|must|might|need|dare)n\'t\b", r"\1 not", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\b(n)'t\b", r" not", sentence, flags=re.IGNORECASE)  # 泛化规则，处理一般的 n't 缩写

    # 对句子进行分词和词性标注
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    processed_words = []

    for word, initial_tag in pos_tags:
        # 跳过标点符号
        if word in string.punctuation:
            continue

        # print(f"Original word: {word}, Original POS: {initial_tag}")  # 输出原始词性

        # 如果是动词，则做 lemma 词性还原
        if initial_tag.startswith('VB'):
            word = lemmatizer.lemmatize(word, pos='v')
            word = stemmer.stem(word)
        
        # 如果是普通名词（NN, NNS），先判断后缀并提取词根
        elif initial_tag in ['NN', 'NNS']:
            # 如果没有指定的后缀，则提取词根
            if not any(word.endswith(suffix) for suffix in noun_suffixes):
                stemmed_word = stemmer.stem(word)
                new_tag = nltk.pos_tag([stemmed_word])[0][1]
                # print(f"Stemmed word: {stemmed_word}, New POS: {new_tag}")  # 输出提取词根后的词性

                # 如果词性不再是名词，则还原为原始单词
                if new_tag not in ['NN', 'NNS']:
                    word = word
                else:
                    word = stemmed_word

        # 如果是复数专有名词（NNPS），还原为单数专有名词（NNP）
        elif initial_tag == 'NNPS':
            word = lemmatizer.lemmatize(word, pos='n')
            new_tag = nltk.pos_tag([word])[0][1]
            # print(f"Lemmatized proper noun: {word}, New POS: {new_tag}")  # 输出还原后的词性

            # 如果还原后不是专有名词，则还原为原始单词
            if new_tag != 'NNP':
                word = word

        # 添加处理后的单词到列表
        processed_words.append(word)
    
    # 返回处理后的句子
    return processed_words

def search_query(index, query):
    # 直接获取语法处理后的查询词数组
    query_words = process_sentence_grammar_2(query)
    
    # 用于存储每个查询词找到的文档集合
    word_doc_sets = []
    
    # 对每个查询词进行检索
    for word in query_words:
        if word in index:
            # 如果词在索引中，存储包含此词的所有 doc_id
            word_doc_sets.append(set(index[word].keys()))
        else:
            # 如果其中一个词不在索引中，则无需继续，直接返回 False
            # print(f"No results found for '{word}'")
            return False
    
    # 取交集：找到包含所有查询词的 doc_id
    common_docs = set.intersection(*word_doc_sets) if word_doc_sets else set()
    
    if not common_docs:
        # 如果没有共同的文档，返回 False
        return False

    # 构建并返回结果，只包含共同 doc_id 的索引行
    results = {}
    for word in query_words:
        if word in index:
            # 获取共同文档的详细信息
            filtered_docs = {doc_id.split('.')[0]: index[word][doc_id] for doc_id in common_docs}
            results[word] = filtered_docs
    
    return results

# rank algorithm
# 计算查询词在文档中的距离 (distance) 和顺序 (order)。
def calculate_distance_order(positions):
    distance = 0
    order = 0
    for i in range(1, len(positions)):
        diff = positions[i] - positions[i - 1]
        distance += abs(diff)
        if diff >= 0:
            order += 1
    return distance, order

# 在查询词位置出现多次的情况下，列出所有可能的词位置组合，找到最佳距离和顺序组合。
def find_best_distance_order(word_positions, result, doc_id):
    # 初始最佳距离无穷大
    best_distance = float('inf')
    # 初始化最佳order
    best_order = 0
    best_combo = {}
    for combo in product(*word_positions.values()):
        # 组合中的位置按词顺序排列
        positions = list(combo)
        # 计算该组合的距离和顺序
        distance, order = calculate_distance_order(combo)
        # 选择距离最小且顺序优先的组合
        if (distance < best_distance) or (distance == best_distance and order > best_order):
            best_distance = distance
            best_order = order
            best_combo = {word: {doc_id: {pos: result[word][doc_id][pos]}} for word, pos in zip(word_positions.keys(), positions)}  # 记录当前最佳组合的位置
            
    return best_distance, best_order, best_combo

def rank_documents(result):
    ranked_results = []
    
    # Step 1: 获取包含所有查询词的共享文档
    # 我们只对包含所有查询词的文档进行排名
    doc_ids = set.intersection(*(set(positions.keys()) for positions in result.values()))
    
    # Step 2: 遍历每个共享文档，计算距离和顺序
    for doc_id in doc_ids:
        # 获取每个查询词在该文档中的所有可能位置
        word_positions = {word: list(positions[doc_id].keys()) for word, positions in result.items()}

        # Step 3: 根据词的出现次数，选择不同的距离计算方式
        if all(len(pos) == 1 for pos in word_positions.values()):
            # 情况 1：如果每个查询词在该文档中只出现一次
            single_positions = {word: pos[0] for word, pos in word_positions.items()}
            distance, order = calculate_distance_order(list(single_positions.values()))
            # 格式化最佳组合为 {word: {doc_id: {position: line_number}}}
            best_combo = {word: {doc_id: {pos: result[word][doc_id][pos]}} for word, pos in single_positions.items()}
        else:
            # 情况 2：如果某个查询词在该文档中出现多次
            # 使用 find_best_distance_order 找到最佳组合
            distance, order, best_combo = find_best_distance_order(word_positions, result, doc_id)

        # 将文档的计算结果添加到排名结果中，包括最佳组合信息
        ranked_results.append((doc_id, distance, order, best_combo))

    # Step 4: 排序结果
    # 排序优先级：距离最小 -> 顺序最大 -> doc_id 数值小
    ranked_results.sort(key=lambda x: (x[1], -x[2], int(x[0])))

    # Step 5: 格式化输出
    final_output = [best_combo for _, _, _, best_combo in ranked_results]
    
    return final_output

def get_line_content(doc_id, line_num, folder_of_documents):
    # file_path = os.path.join(folder_of_documents, doc_id)
    file_path = os.path.join(folder_of_documents, f"{doc_id}.txt")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for current_line_num, line in enumerate(file, start=1):
            if current_line_num == line_num:
                return line.strip()  # 返回去除首尾空格的行内容
               
    return None  # 如果未找到行，返回 None

def generate_substrings(word):
    return [word[:i] for i in range(len(word), 0, -1)]

def pre_process_query(query, dictionary):
    processed_query = []
    for word in query.split():
        substrings = generate_substrings(word)
        found_match = False
        for subword in substrings:
            matches = difflib.get_close_matches(subword, dictionary, n=1, cutoff=0.9)
            if matches:
                processed_query.append(matches[0])
                found_match = True
                break
        if not found_match:
            processed_query.append(word)  # 保持原样
    return ' '.join(processed_query)

# spelling correction
def spelling_correction(query, index):
    # 初始化用于存储拼写纠正候选词的二维数组
    corrected_candidates = []
    
    for word in query:
        # 根据单词长度设定参数
        if len(word) >= 8:
            n = 3
            cutoff = max(0.0, min(1.0, (len(word) - 2) / len(word))) 
        else:
            n = 5
            cutoff = max(0.0, min(1.0, (len(word) - 3) / len(word)))
        
        # 获取拼写纠正的候选词
        matches = difflib.get_close_matches(word, index.keys(), n=n, cutoff=cutoff)
        corrected_candidates.append(matches or [word])  # 如果没有匹配项，保留原词
    
    return corrected_candidates

# get the nice combination when "spelling correction" happened
def get_best_correction_combination(corrected_candidates, index):
    all_combinations = list(itertools.product(*corrected_candidates))
    best_combination = None
    best_rank = float('inf')
    best_ranked_results = None  # 用于保存最佳组合的查询结果

    for combination in all_combinations:
        result = search_query(index, " ".join(combination))
        if result:
            ranked_results = rank_documents(result)
            rank_score = len(ranked_results)

            if rank_score < best_rank:
                best_rank = rank_score
                best_combination = combination
                best_ranked_results = ranked_results  # 保存最佳结果

    return best_combination, best_rank, best_ranked_results

def output_ranked_results(ranked_results, starts_with_symbol, index_folder):
    # 输出格式化结果
    if starts_with_symbol:
        for item in ranked_results:
            doc_ids = list(item.values())[0].keys()
            for doc_id in doc_ids:
                print(doc_id)
    else:
        all_lines = []  # 用于存储行号、内容和文档ID的元组
        seen_lines = set()  # 用于去重

        # 保留 doc_id 的顺序
        for item in ranked_results:
            doc_ids = list(item.values())[0].keys()  # 获取文档ID集合
            for doc_id in doc_ids:
                # 存储当前 doc_id 的所有行号
                doc_lines = []
                for word, doc_info in item.items():
                    line_num = list(doc_info[doc_id].values())[0]  # 获取行号
                    line_content = get_line_content(doc_id, line_num, index_folder)
                    
                    # 确保不重复添加相同行号的内容
                    if (line_num, line_content) not in seen_lines:
                        seen_lines.add((line_num, line_content))
                        doc_lines.append((line_num, line_content))

                # 对当前 doc_id 的行号排序
                doc_lines.sort(key=lambda x: x[0])

                # 将排序后的行内容添加到 all_lines，同时保留 doc_id
                for line_num, line_content in doc_lines:
                    all_lines.append((line_num, line_content, doc_id))

        # 输出结果
        last_doc_id = None
        for line_num, line_content, doc_id in all_lines:
            # 如果是新的 doc_id，则打印 > doc_id
            if doc_id != last_doc_id:
                print(f"> {doc_id}")
                last_doc_id = doc_id
            print(" " + line_content)
        

def process_query(index, query, index_folder):
    # 检查查询是否以 "> " 开头
    starts_with_symbol = not query.startswith("> ")
    if starts_with_symbol:
        query = query.lstrip("> ").strip()  # 去掉开头的"> "，并去除首尾空格

    # 用 search_query 处理查询，得到包含交集的结果
    result = search_query(index, query)
    
    if not result:
        # print("Spelling Problem")
        # 进行拼写纠正
        # 将 query 转为小写并进行预处理
        query = query.lower()
        dictionary = list(index.keys())  # 假设字典是索引的键值集合
        query = pre_process_query(query, dictionary)
        corrected_candidates = spelling_correction(query.split(), index)
        
        # 获取最佳拼写纠正组合
        best_combination, best_rank, best_ranked_results= get_best_correction_combination(corrected_candidates, index)
        
        if best_combination:
            # print("Best correction:", " ".join(best_combination))
            output_ranked_results(best_ranked_results, starts_with_symbol, index_folder)  # 直接使用最佳结果
        else:
            print("No suitable correction found.")
    else:
        # 有结果的情况下直接进行排名和输出
        ranked_results = rank_documents(result)
        # print(ranked_results)
        output_ranked_results(ranked_results, starts_with_symbol, index_folder)


# python3 search.py ./MyTestIndex
def main():
    # 确保正确的命令行参数数量
    if len(sys.argv) < 2:
        print("Usage: python3 search.py [index_folder]")
        sys.exit(1)

    # 获取 index_folder
    index_folder = sys.argv[1]
    index = load_index(index_folder)

    # 检查是否有文件重定向输入
    if not sys.stdin.isatty():
        # 文件重定向的输入
        for query in sys.stdin:
            query = query.strip()
            # print(query)
            query = query.strip()
            if not query:
                continue  # 跳过空行
            process_query(index, query, index_folder)
    else:
        # 手动输入模式
        while True:
            query = input().strip()
            if not query:  # 如果输入为空，结束循环
                # print("Exiting search.")
                break
            process_query(index, query, index_folder)

if __name__ == "__main__":
    main()
    
# common where modern