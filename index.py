import re
import string
import nltk
import os
import shutil
from collections import defaultdict
nltk.download('averaged_perceptron_tagger', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)
nltk.download('punkt', quiet = True)
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# 文件格式预处理
def process_text(text):
    # 1. 将所有字母转换为小写
    text = text.lower()

    # 2. 省略缩写中的点，如u.s变成us
    text = re.sub(r'\b([a-z])\.(?=[a-z]\b)', r'\1', text)

    # 3. 去掉数字中的逗号，如100,000变成100000
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)

    # 4. 去掉小数（跳过小数），例如 1.234 被忽略
    text = re.sub(r'\b\d+\.\d+\b', '', text)

    # 5. 去掉分隔符，如1989/11变成“1989” “11”
    text = re.sub(r'(\d{4})/(\d{2})', r'\1 \2', text)

    # 6. 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 将文本拆分为单词列表
    words = text.split()

    # 过滤掉空的项（例如因为小数被替换为空字符后产生的空格）
    processed_words = [word for word in words if word]

    return processed_words

# sample_text_process = "The U.S. has a population of 100,000. In 1989/11, the GDP was 123.45 billion dollars!"
# processed_text = process_text(sample_text_process)
# print(processed_text)

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
    return " ".join(processed_words)

# 示例使用
# sentence = "The cat's toys, books, U.S and bowls are new!"
# processed_sentence = process_sentence_grammar_2(sentence)
# print("Processed sentence:", processed_sentence)

def build_index(folder_of_documents, folder_of_indexes):
    # structure:{word: {docID1:{position1: lineNumber1, position2: lineNumber2, ...}, docID2:{position1: lineNumber1, position2: lineNumber2, ...}}} 
    index = defaultdict(lambda: defaultdict(dict))
    
    # 1. 通过filename遍历文件夹。
    for filename in os.listdir(folder_of_documents):
        doc_id = os.path.splitext(filename)[0]  # 去掉扩展名
        file_path = os.path.join(folder_of_documents, filename)
        
        # 在循环之前设计一个常量, 索引要+这个常量才是他在全文的索引，并且获取数组长度累加到常量，方便下一组得到全文索引（position）
        position_in_document = 0
        # 当前行号
        line_number = 1
        # 暂存器，存放未结束的句子片段
        buffer = ""
        previous_buffer_len = 0

        # 打开并读取文档内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        # 2. 然后通过换行号为分割遍历文章，从1开始，每轮结束行号+1来获得和更新行号。
            for line in file:
                line = line.strip()  # 去掉行首行尾的空白
                if not line:
                    line_number += 1
                    continue  # 跳过空行
                
                # buffer与下一行的第一句一起处理。buffer中的行号为“目前行号-1”，buffer中单词的索引为“索引记录常量 - buffer.len + 单词在buffer的index"
                if buffer:
                    line = buffer + " " + line
                    buffer_line_number = line_number - 1  # 合并时减 1 行号
                    buffer_len = len(process_text(buffer))
                    buffer = ""
                else:
                    buffer_len = 0
                    buffer_line_number = line_number
                
                # 3. 检查最后一句是否是完整句子（以‘.','!','?'结尾）如果不是, 进行格式处理，放入buffer。
                # 4. 接下来进行文本处理，首先做语法处理，然后再做格式处理，生成[word1, word2, word3...]的格式。
                # 5. 遍历数组，提取单词并记下索引
                # 查找最后一个结束标点的位置
                last_punctuation_index = max(line.rfind('.'), line.rfind('!'), line.rfind('?'))
                
                if last_punctuation_index != -1:
                    # 有结束标点，将结束标点之前的部分作为完整句子处理
                    complete_sentence = line[:last_punctuation_index + 1]
                    processed_sentence = process_sentence_grammar_2(complete_sentence)
                    words = process_text(processed_sentence)

                    # 遍历当前句子的单词并记录索引和行号
                    for word_index, word in enumerate(words):
                        # 判断行号：如果是 buffer 部分的单词，使用 buffer_line_number，否则使用 line_number
                        if word_index < previous_buffer_len:
                            index[word][doc_id][position_in_document + word_index] = buffer_line_number - 1
                        # 接着 current_buffer_len 个使用 buffer_line_number
                        elif word_index < buffer_len:
                            index[word][doc_id][position_in_document + word_index] = buffer_line_number
                        else:
                            # 其余的使用当前的 line_number
                            index[word][doc_id][position_in_document + word_index] = line_number
                            
                    # 更新全文位置常量
                    position_in_document += len(words)

                    # 将结束标点之后的部分存入缓冲区
                    buffer = line[last_punctuation_index + 1:].strip()
                    
                    previous_buffer_len = 0
                else:
                    # 当前行没有结束标点，将其存入缓冲器
                    buffer = line
                    # 更新 position_in_document 以反映 buffer 中单词的数量
                    previous_buffer_len = buffer_len

                # 行号自增
                line_number += 1
                
            # 循环结束之后，处理文件结尾残留在缓冲器中的内容（仅做格式处理，不做语法处理）
            if buffer:
                buffer_processed_words = process_text(buffer)
                for word_index, word in enumerate(buffer_processed_words):
                    # 使用 buffer_line_number - 1 记录缓冲区的行号
                    if word_index < previous_buffer_len:
                        index[word][doc_id][position_in_document + word_index] = buffer_line_number - 1
                    else:
                        index[word][doc_id][position_in_document + word_index] = buffer_line_number
        # 复制txt文件到索引文件夹
        if not os.path.exists(folder_of_indexes):
            os.makedirs(folder_of_indexes)
                
        shutil.copy(file_path, os.path.join(folder_of_indexes, filename))        
    # 文件处理完毕后，创建索引文件夹并保存索引到文件
    # if not os.path.exists(folder_of_indexes):
    #     os.makedirs(folder_of_indexes)

    index_file_path = os.path.join(folder_of_indexes, 'index.txt')
    with open(index_file_path, 'w', encoding='utf-8') as index_file:
        for word, doc_info in index.items():
            index_file.write(f"{word}: {dict(doc_info)}\n")

    # print(f"Index built successfully and saved to {index_file_path}")
    
# # 使用示例
# folder_of_documents = "/Users/bella_pong/Desktop/Ranked_Retrieval/collection"    # 文档文件夹路径
# folder_of_indexes = "./MyTestIndex"        # 索引文件夹路径

# build_index(folder_of_documents, folder_of_indexes)
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 index.py [folder-of-documents] [folder-of-indexes]")
        sys.exit(1)

    folder_of_documents = sys.argv[1]
    folder_of_indexes = sys.argv[2]
    build_index(folder_of_documents, folder_of_indexes)

# python3 index.py /Users/bella_pong/Desktop/Ranked_Retrieval/collection ./MyTestIndex