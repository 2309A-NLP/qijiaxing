# -*- coding: utf-8 -*-
"""
BM25关键词检索模块（增强版）
功能：基于BM25算法的关键词检索，支持中文分词、布尔查询、短语匹配、模糊匹配

模块职责：
1. BM25Retriever - BM25检索器，实现经典的BM25排序算法
2. HybridRetriever - 混合检索器，结合向量检索和BM25

增强功能（v2）：
- 布尔查询：支持 AND、OR、NOT 运算符
- 短语匹配：支持双引号精确短语匹配
- 模糊匹配：支持部分匹配和编辑距离容错

BM25算法原理：
BM25（Best Matching 25）是一种经典的信息检索排序算法，
用于计算文档与查询之间的相关性得分。
核心公式：score(D,Q) = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1*(1-b+b*|D|/avgdl))

优势：
- 对词频进行饱和处理，避免长文档过度优势
- 考虑文档长度归一化
- 计算效率高
"""

# 【作用】导入数学库，用于计算对数（IDF公式中用到了log运算）
# 【原理】math.log()用于计算自然对数，BM25的IDF公式中包含对数运算
import math
# 【作用】导入正则表达式模块，用于清理文本（去除非汉字/英文/数字的字符）
# 【原理】re.sub()用正则替换掉所有干扰符号，让jieba分词更准确
import re
# 【作用】导入类型注解，让函数签名更清晰
# 【原理】Tuple[Dict, float] 表示函数返回（文档字典，得分）的元组
from typing import List, Dict, Tuple, Optional, Set
# 【作用】从collections导入Counter类，用于统计词频
# 【原理】Counter是Python内置的高效计数器，接收列表返回{元素:出现次数}字典
from collections import Counter
# 【作用】导入jieba中文分词库
# 【原理】jieba是Python最流行的中文分词库，支持精确模式和搜索引擎模式
import jieba


class QueryParser:
    """
    查询解析器
    
    支持三种查询模式：
    1. 布尔查询：AND、OR、NOT 运算符
       示例：'财务 AND 报告'、'营收 OR 收入'、'报告 NOT 审计'
    2. 短语匹配：双引号精确匹配
       示例：'"年度财务报告"'、'"营业收入"'
    3. 模糊匹配：默认模式，支持部分匹配
       示例：'财务报告'（会匹配包含'财务'或'报告'的文档）
    
    返回类型：
      parse(query) → ParsedQuery
      ParsedQuery 包含：terms, phrase_terms, boolean_op, is_phrase_query
    
    边界情况：
      - 空查询：返回空ParsedQuery
      - 混合模式：短语+布尔组合（如 '"财务报告" AND 审计'）
      - 语法错误：降级为普通模糊匹配
    
    面试官可能问：
      Q: 为什么要单独做查询解析器？
      A: 不同查询类型需要不同的检索策略：
         - 布尔查询需要集合运算（交集/并集/差集）
         - 短语查询需要位置索引（词序连续性）
         - 模糊查询需要编辑距离或子串匹配
         统一解析后，检索器可以根据类型选择最优策略。
    """
    
    def __init__(self):
        # 【作用】定义布尔运算符集合
        self.boolean_ops = {'AND', 'OR', 'NOT'}
        # 【作用】定义短语匹配的正则表达式
        # 【原理】匹配双引号内的内容，支持中英文
        self.phrase_pattern = re.compile(r'"([^"]+)"')
    
    def parse(self, query: str) -> 'ParsedQuery':
        """
        解析查询字符串
        
        Args:
            query: 原始查询字符串
            
        Returns:
            ParsedQuery 对象，包含解析后的查询组件
        """
        if not query or not query.strip():
            return ParsedQuery()
        
        query = query.strip()
        
        # 【作用】检测是否为纯短语查询
        # 【原理】如果整个查询被双引号包裹，且没有布尔运算符
        if query.startswith('"') and query.endswith('"') and len(query) > 2:
            phrase = query[1:-1].strip()
            if phrase:
                return ParsedQuery(
                    terms=[],
                    phrase_terms=[phrase],
                    boolean_op=None,
                    is_phrase_query=True
                )
        
        # 【作用】提取所有短语项
        # 【原理】用正则找到所有双引号包裹的内容
        phrase_terms = self.phrase_pattern.findall(query)
        
        # 【作用】移除短语部分，处理剩余的布尔查询
        # 【原理】用正则替换掉短语，保留布尔运算符和普通词
        remaining = self.phrase_pattern.sub('', query).strip()
        
        # 【作用】检测布尔运算符
        # 【原理】检查是否包含 AND、OR、NOT
        boolean_op = None
        terms = []
        
        if ' AND ' in remaining:
            boolean_op = 'AND'
            parts = remaining.split(' AND ')
            terms = [p.strip() for p in parts if p.strip()]
        elif ' OR ' in remaining:
            boolean_op = 'OR'
            parts = remaining.split(' OR ')
            terms = [p.strip() for p in parts if p.strip()]
        elif remaining.startswith('NOT '):
            boolean_op = 'NOT'
            terms = [remaining[4:].strip()]
        else:
            # 【作用】普通查询，按空格分词
            terms = remaining.split()
        
        # 【作用】过滤空词
        terms = [t for t in terms if t]
        
        return ParsedQuery(
            terms=terms,
            phrase_terms=phrase_terms,
            boolean_op=boolean_op,
            is_phrase_query=len(phrase_terms) > 0 and len(terms) == 0
        )


class ParsedQuery:
    """
    解析后的查询对象
    
    属性：
      terms: 普通查询词列表
      phrase_terms: 短语查询词列表（需要精确匹配）
      boolean_op: 布尔运算符（AND/OR/NOT/None）
      is_phrase_query: 是否为纯短语查询
    """
    
    def __init__(
        self,
        terms: List[str] = None,
        phrase_terms: List[str] = None,
        boolean_op: Optional[str] = None,
        is_phrase_query: bool = False
    ):
        self.terms: List[str] = terms if terms is not None else []
        self.phrase_terms: List[str] = phrase_terms if phrase_terms is not None else []
        self.boolean_op = boolean_op
        self.is_phrase_query = is_phrase_query
    
    def is_empty(self) -> bool:
        """检查查询是否为空"""
        return len(self.terms) == 0 and len(self.phrase_terms) == 0
    
    def get_all_terms(self) -> List[str]:
        """获取所有查询词（包括短语分词后的结果）"""
        all_terms = list(self.terms)
        for phrase in self.phrase_terms:
            # 【作用】对短语进行分词，用于BM25计算
            all_terms.extend(phrase.split())
        return all_terms


class BM25Retriever:
    """
    BM25检索器类（增强版）

    实现经典的BM25排序算法，用于关键词检索。
    BM25是TF-IDF的改进版本，解决了词频饱和和文档长度偏差问题。

    增强功能：
    - 布尔查询支持（AND/OR/NOT）
    - 短语匹配（精确匹配）
    - 模糊匹配（部分匹配）

    核心参数：
    - k1: 控制词频饱和度，值越大词频影响越大（通常1.2-2.0）
    - b: 控制文档长度归一化，值越大长文档惩罚越重（通常0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        作用：初始化BM25检索器

        Args:
            k1: BM25参数，控制词频饱和度。
                - 值越小，对高频词的惩罚越严格
                - 值越大，词频的影响力越大
                - 推荐范围：1.2-2.0
            b: BM25参数，控制文档长度归一化程度。
                - 值越大，对长文档的惩罚越严重
                - 推荐范围：0.5-0.75
        """
        # 【作用】保存词频饱和参数k1
        self.k1 = k1
        # 【作用】保存长度归一化参数b
        self.b = b

        # 【作用】创建空的数据结构，等待build_index()填充
        self.documents: List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        
        # 【作用】查询解析器实例
        self.query_parser = QueryParser()

    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """
        作用：构建BM25索引

        构建索引是检索前的必要步骤，会：
        1. 对所有文档进行分词
        2. 统计每个词的文档频率
        3. 计算平均文档长度

        Args:
            documents: 文档列表，每个文档是字典类型
            text_field: 要索引的文本字段名，默认为"content"
        """
        self.documents = documents
        self.N = len(documents)

        self.tokenized_docs = []
        self.doc_lengths = []
        term_doc_count: Dict[str, int] = {}

        for doc in documents:
            text = doc.get(text_field, "")
            tokens = self._tokenize(text)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

            unique_terms = set(tokens)
            for term in unique_terms:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        self.doc_freqs = term_doc_count
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

    def _tokenize(self, text: str) -> List[str]:
        """
        作用：中文分词

        使用jieba分词引擎对中文文本进行分词，
        并过滤停用词和短词。

        Args:
            text: 输入文本

        Returns:
            分词结果列表，如 ["公司", "财务", "数据"]
        """
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        tokens = list(jieba.cut(text))

        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }

        tokens = [
            t.strip()
            for t in tokens
            if len(t.strip()) > 1
            and t.strip() not in stop_words
        ]

        return tokens

    def _calc_idf(self, term: str) -> float:
        """
        作用：计算IDF（逆文档频率）

        IDF公式：IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        Args:
            term: 查询词

        Returns:
            IDF值
        """
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _calc_bm25_score(self, doc_idx: int, query_terms: List[str]) -> float:
        """
        作用：计算单个文档的BM25得分

        BM25公式：
        score = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1*(1-b+b*|D|/avgdl))

        Args:
            doc_idx: 文档索引
            query_terms: 查询词列表

        Returns:
            BM25得分（越高越相关）
        """
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        term_freq = Counter(doc_tokens)

        for term in query_terms:
            if term not in term_freq:
                continue

            idf = self._calc_idf(term)
            tf = term_freq[term]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def _check_phrase_match(self, doc_idx: int, phrase: str) -> bool:
        """
        检查文档是否包含精确短语

        Args:
            doc_idx: 文档索引
            phrase: 要匹配的短语

        Returns:
            True 如果文档包含该短语
        """
        # 【作用】获取原始文档文本
        doc = self.documents[doc_idx]
        text = doc.get("content", doc.get("text", ""))
        
        # 【作用】检查短语是否在文本中
        return phrase in text

    def _check_fuzzy_match(self, doc_idx: int, term: str, threshold: float = 0.6) -> Tuple[bool, float]:
        """
        模糊匹配检查

        Args:
            doc_idx: 文档索引
            term: 查询词
            threshold: 匹配阈值（0-1）

        Returns:
            (是否匹配, 匹配分数)
        """
        doc_tokens = self.tokenized_docs[doc_idx]
        
        # 【作用】检查完全匹配
        if term in doc_tokens:
            return True, 1.0
        
        # 【作用】检查子串匹配
        for token in doc_tokens:
            if term in token or token in term:
                return True, 0.8
        
        # 【作用】检查编辑距离（仅对长度>2的词）
        if len(term) > 2:
            for token in doc_tokens:
                if len(token) > 2:
                    # 【作用】计算编辑距离
                    distance = self._levenshtein_distance(term, token)
                    max_len = max(len(term), len(token))
                    similarity = 1 - (distance / max_len)
                    if similarity >= threshold:
                        return True, similarity
        
        return False, 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离（Levenshtein Distance）

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def search(
        self, 
        query: str, 
        top_k: int = 10,
        enable_boolean: bool = True,
        enable_phrase: bool = True,
        enable_fuzzy: bool = True
    ) -> List[Tuple[Dict, float]]:
        """
        作用：检索相关文档（增强版）

        支持三种查询模式：
        1. 布尔查询：AND、OR、NOT
        2. 短语匹配：双引号精确匹配
        3. 模糊匹配：部分匹配和编辑距离

        Args:
            query: 查询字符串
            top_k: 返回前k个最相关的文档
            enable_boolean: 是否启用布尔查询
            enable_phrase: 是否启用短语匹配
            enable_fuzzy: 是否启用模糊匹配

        Returns:
            [(文档, BM25得分), ...]，按得分降序排列
        """
        if not self.documents:
            return []

        # 【作用】解析查询
        parsed = self.query_parser.parse(query)
        
        if parsed.is_empty():
            return []

        # 【作用】根据查询类型选择检索策略
        
        # 【情况1】纯短语查询
        if parsed.is_phrase_query and enable_phrase:
            return self._search_phrase(parsed.phrase_terms[0], top_k)
        
        # 【情况2】布尔查询
        if parsed.boolean_op and enable_boolean:
            return self._search_boolean(parsed, top_k, enable_fuzzy)
        
        # 【情况3】混合查询（短语+普通词）
        if parsed.phrase_terms and parsed.terms:
            return self._search_mixed(parsed, top_k, enable_fuzzy)
        
        # 【情况4】普通模糊查询
        if enable_fuzzy:
            return self._search_fuzzy(parsed.terms, top_k)
        
        # 【情况5】标准BM25查询
        return self._search_standard(parsed.get_all_terms(), top_k)

    def _search_standard(self, query_terms: List[str], top_k: int) -> List[Tuple[Dict, float]]:
        """标准BM25检索"""
        if not query_terms:
            return []

        scores = []
        for idx in range(self.N):
            score = self._calc_bm25_score(idx, query_terms)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:top_k]]

    def _search_phrase(self, phrase: str, top_k: int) -> List[Tuple[Dict, float]]:
        """
        短语精确匹配检索

        Args:
            phrase: 要匹配的短语
            top_k: 返回数量

        Returns:
            匹配的文档列表
        """
        results = []
        
        # 【作用】对短语分词，用于BM25打分
        phrase_terms = self._tokenize(phrase)
        
        for idx in range(self.N):
            # 【作用】检查是否包含精确短语
            if self._check_phrase_match(idx, phrase):
                # 【作用】计算BM25得分（使用短语分词）
                score = self._calc_bm25_score(idx, phrase_terms) if phrase_terms else 1.0
                results.append((idx, score * 1.5))  # 【作用】短语匹配加权
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def _search_boolean(
        self, 
        parsed: ParsedQuery, 
        top_k: int,
        enable_fuzzy: bool
    ) -> List[Tuple[Dict, float]]:
        """
        布尔查询检索

        Args:
            parsed: 解析后的查询
            top_k: 返回数量
            enable_fuzzy: 是否启用模糊匹配

        Returns:
            匹配的文档列表
        """
        if parsed.boolean_op == 'AND':
            # 【作用】AND：所有词都必须出现
            return self._search_and(parsed.terms, parsed.phrase_terms, top_k, enable_fuzzy)
        elif parsed.boolean_op == 'OR':
            # 【作用】OR：任一词出现即可
            return self._search_or(parsed.terms, parsed.phrase_terms, top_k, enable_fuzzy)
        elif parsed.boolean_op == 'NOT':
            # 【作用】NOT：排除包含指定词的文档
            return self._search_not(parsed.terms, top_k, enable_fuzzy)
        
        return []

    def _search_and(
        self, 
        terms: List[str], 
        phrase_terms: List[str],
        top_k: int,
        enable_fuzzy: bool
    ) -> List[Tuple[Dict, float]]:
        """AND查询：所有条件都必须满足"""
        results = []
        
        # 【作用】对普通词分词
        all_term_tokens = []
        for term in terms:
            all_term_tokens.extend(self._tokenize(term))
        
        for idx in range(self.N):
            doc_tokens = set(self.tokenized_docs[idx])
            
            # 【作用】检查所有普通词是否都出现
            terms_match = True
            if all_term_tokens:
                if enable_fuzzy:
                    # 【作用】模糊匹配模式
                    for term in all_term_tokens:
                        matched, _ = self._check_fuzzy_match(idx, term)
                        if not matched:
                            terms_match = False
                            break
                else:
                    # 【作用】精确匹配模式
                    for term in all_term_tokens:
                        if term not in doc_tokens:
                            terms_match = False
                            break
            
            if not terms_match:
                continue
            
            # 【作用】检查所有短语是否都出现
            phrases_match = True
            for phrase in phrase_terms:
                if not self._check_phrase_match(idx, phrase):
                    phrases_match = False
                    break
            
            if not phrases_match:
                continue
            
            # 【作用】计算BM25得分
            score = self._calc_bm25_score(idx, all_term_tokens) if all_term_tokens else 1.0
            results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def _search_or(
        self, 
        terms: List[str], 
        phrase_terms: List[str],
        top_k: int,
        enable_fuzzy: bool
    ) -> List[Tuple[Dict, float]]:
        """OR查询：任一条件满足即可"""
        results = []
        
        all_term_tokens = []
        for term in terms:
            all_term_tokens.extend(self._tokenize(term))
        
        for idx in range(self.N):
            doc_tokens = set(self.tokenized_docs[idx])
            
            # 【作用】检查是否有任一词出现
            any_term_match = False
            if all_term_tokens:
                if enable_fuzzy:
                    for term in all_term_tokens:
                        matched, _ = self._check_fuzzy_match(idx, term)
                        if matched:
                            any_term_match = True
                            break
                else:
                    for term in all_term_tokens:
                        if term in doc_tokens:
                            any_term_match = True
                            break
            
            # 【作用】检查是否有任一短语出现
            any_phrase_match = False
            for phrase in phrase_terms:
                if self._check_phrase_match(idx, phrase):
                    any_phrase_match = True
                    break
            
            # 【作用】任一条件满足即可
            if any_term_match or any_phrase_match:
                score = self._calc_bm25_score(idx, all_term_tokens) if all_term_tokens else 1.0
                results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def _search_not(
        self, 
        exclude_terms: List[str], 
        top_k: int,
        enable_fuzzy: bool
    ) -> List[Tuple[Dict, float]]:
        """NOT查询：排除包含指定词的文档"""
        results = []
        
        exclude_tokens = []
        for term in exclude_terms:
            exclude_tokens.extend(self._tokenize(term))
        
        for idx in range(self.N):
            doc_tokens = set(self.tokenized_docs[idx])
            
            # 【作用】检查是否包含排除词
            should_exclude = False
            if enable_fuzzy:
                for term in exclude_tokens:
                    matched, _ = self._check_fuzzy_match(idx, term)
                    if matched:
                        should_exclude = True
                        break
            else:
                for term in exclude_tokens:
                    if term in doc_tokens:
                        should_exclude = True
                        break
            
            # 【作用】不包含排除词的文档才保留
            if not should_exclude:
                # 【作用】使用排除词的反面作为相关性（文档越长可能越相关）
                score = 1.0 / (self.doc_lengths[idx] + 1)
                results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def _search_mixed(
        self, 
        parsed: ParsedQuery, 
        top_k: int,
        enable_fuzzy: bool
    ) -> List[Tuple[Dict, float]]:
        """混合查询：短语+普通词"""
        results = []
        
        term_tokens = []
        for term in parsed.terms:
            term_tokens.extend(self._tokenize(term))
        
        for idx in range(self.N):
            doc_tokens = set(self.tokenized_docs[idx])
            
            # 【作用】检查短语匹配
            phrase_matched = True
            for phrase in parsed.phrase_terms:
                if not self._check_phrase_match(idx, phrase):
                    phrase_matched = False
                    break
            
            if not phrase_matched:
                continue
            
            # 【作用】检查普通词匹配
            terms_matched = True
            if term_tokens:
                if enable_fuzzy:
                    for term in term_tokens:
                        matched, _ = self._check_fuzzy_match(idx, term)
                        if not matched:
                            terms_matched = False
                            break
                else:
                    for term in term_tokens:
                        if term not in doc_tokens:
                            terms_matched = False
                            break
            
            if not terms_matched:
                continue
            
            # 【作用】计算综合得分
            score = self._calc_bm25_score(idx, term_tokens) if term_tokens else 1.0
            score *= 1.3  # 【作用】混合匹配加权
            results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def _search_fuzzy(
        self, 
        terms: List[str], 
        top_k: int,
        fuzzy_threshold: float = 0.6
    ) -> List[Tuple[Dict, float]]:
        """
        模糊匹配检索

        Args:
            terms: 查询词列表
            top_k: 返回数量
            fuzzy_threshold: 模糊匹配阈值

        Returns:
            匹配的文档列表
        """
        results = []
        
        term_tokens = []
        for term in terms:
            term_tokens.extend(self._tokenize(term))
        
        if not term_tokens:
            return []
        
        for idx in range(self.N):
            doc_tokens = self.tokenized_docs[idx]
            total_score = 0.0
            matched_count = 0
            
            for term in term_tokens:
                # 【作用】精确匹配
                if term in doc_tokens:
                    score = self._calc_bm25_score(idx, [term])
                    total_score += score
                    matched_count += 1
                    continue
                
                # 【作用】子串匹配
                substring_matched = False
                for token in doc_tokens:
                    if term in token or token in term:
                        score = self._calc_bm25_score(idx, [term]) * 0.8
                        total_score += score
                        matched_count += 1
                        substring_matched = True
                        break
                
                if substring_matched:
                    continue
                
                # 【作用】编辑距离匹配
                if len(term) > 2:
                    best_similarity = 0
                    for token in doc_tokens:
                        if len(token) > 2:
                            distance = self._levenshtein_distance(term, token)
                            max_len = max(len(term), len(token))
                            similarity = 1 - (distance / max_len)
                            best_similarity = max(best_similarity, similarity)
                    
                    if best_similarity >= fuzzy_threshold:
                        score = self._calc_bm25_score(idx, [term]) * best_similarity
                        total_score += score
                        matched_count += 1
            
            if matched_count > 0:
                # 【作用】平均得分
                avg_score = total_score / len(term_tokens)
                results.append((idx, avg_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in results[:top_k]]

    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 10,
        enable_boolean: bool = True,
        enable_phrase: bool = True,
        enable_fuzzy: bool = True
    ) -> List[List[Tuple[Dict, float]]]:
        """
        作用：批量检索

        Args:
            queries: 查询列表
            top_k: 每个查询返回前k个结果
            enable_boolean: 是否启用布尔查询
            enable_phrase: 是否启用短语匹配
            enable_fuzzy: 是否启用模糊匹配

        Returns:
            每个查询的检索结果列表
        """
        return [
            self.search(q, top_k, enable_boolean, enable_phrase, enable_fuzzy) 
            for q in queries
        ]


class HybridRetriever:
    """
    作用：混合检索器（增强版）
    原理：结合向量检索（语义相似度）和BM25（关键词匹配）的优点，
          使用RRF（Reciprocal Rank Fusion）算法融合两种检索结果。
    
    增强功能：
    - 支持布尔查询、短语匹配、模糊匹配
    - 可配置的检索策略
    - 动态权重调整接口
    """

    def __init__(
        self, 
        vector_weight: float = 0.7, 
        bm25_weight: float = 0.3,
        enable_boolean: bool = True,
        enable_phrase: bool = True,
        enable_fuzzy: bool = True
    ):
        """
        作用：初始化混合检索器

        Args:
            vector_weight: 向量检索的权重（默认0.7）
            bm25_weight: BM25检索的权重（默认0.3）
            enable_boolean: 是否启用布尔查询
            enable_phrase: 是否启用短语匹配
            enable_fuzzy: 是否启用模糊匹配
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_boolean = enable_boolean
        self.enable_phrase = enable_phrase
        self.enable_fuzzy = enable_fuzzy
        self.bm25 = BM25Retriever()

    def update_weights(self, vector_weight: float, bm25_weight: float):
        """
        动态更新检索权重
        
        Args:
            vector_weight: 新的向量检索权重
            bm25_weight: 新的BM25检索权重
            
        权重会自动归一化，确保总和为1
        """
        # 【作用】权重归一化
        total = vector_weight + bm25_weight
        if total > 0:
            self.vector_weight = vector_weight / total
            self.bm25_weight = bm25_weight / total
        else:
            # 【作用】默认权重
            self.vector_weight = 0.7
            self.bm25_weight = 0.3

    def update_search_options(
        self,
        enable_boolean: Optional[bool] = None,
        enable_phrase: Optional[bool] = None,
        enable_fuzzy: Optional[bool] = None
    ):
        """
        动态更新检索选项
        
        Args:
            enable_boolean: 是否启用布尔查询
            enable_phrase: 是否启用短语匹配
            enable_fuzzy: 是否启用模糊匹配
        """
        if enable_boolean is not None:
            self.enable_boolean = enable_boolean
        if enable_phrase is not None:
            self.enable_phrase = enable_phrase
        if enable_fuzzy is not None:
            self.enable_fuzzy = enable_fuzzy

    def get_config(self) -> Dict:
        """
        获取当前配置
        
        Returns:
            配置字典
        """
        return {
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "enable_boolean": self.enable_boolean,
            "enable_phrase": self.enable_phrase,
            "enable_fuzzy": self.enable_fuzzy
        }

    def build_index(self, documents: List[Dict], text_field: str = "content"):
        """构建BM25索引"""
        self.bm25.build_index(documents, text_field)

    def search(
        self,
        query: str,
        vector_results: List[Tuple[Dict, float]],
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        作用：混合检索（增强版）
        
        Args:
            query: 查询字符串（支持布尔查询、短语匹配）
            vector_results: 向量检索结果
            top_k: 返回数量
            
        Returns:
            融合后的结果
        """
        # 【作用】使用增强版BM25检索
        bm25_results = self.bm25.search(
            query, 
            top_k=top_k * 2,
            enable_boolean=self.enable_boolean,
            enable_phrase=self.enable_phrase,
            enable_fuzzy=self.enable_fuzzy
        )

        # 【作用】RRF融合
        fused_scores = self._rrf_fuse(vector_results, bm25_results)

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        return fused_scores[:top_k]

    def _rrf_fuse(
        self,
        vector_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        RRF（倒数排名融合）算法
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            k: RRF参数（默认60）
            
        Returns:
            融合后的结果
        """
        doc_map: Dict[str, Dict] = {}
        doc_scores: Dict[str, float] = {}

        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.vector_weight / (k + rank)

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + self.bm25_weight / (k + rank)

        results = []
        for doc_id, fused_score in doc_scores.items():
            doc = doc_map[doc_id]
            results.append((doc, fused_score))

        return results


# 【作用】全局混合检索器实例（单例模式）
_hybrid_retriever = None


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """
    作用：获取混合检索器实例（单例）

    Returns:
        HybridRetriever实例，如果初始化失败返回None
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        try:
            _hybrid_retriever = HybridRetriever()
        except Exception:
            return None
    return _hybrid_retriever
