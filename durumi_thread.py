import csv
import threading
from queue import Queue
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm 


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ver = '0.2'
DEBUG = False


durumi_art = """
DDDD   U   U  RRRR   U   U  M   M  III      V   V  EEEE  RRRR      0         222
D   D  U   U  R   R  U   U  MM MM   I       V   V  E     R   R    0 0       2   2
D   D  U   U  RRRR   U   U  M M M   I        V V   EEE   RRRR    0   0        2
D   D  U   U  R R    U   U  M   M   I        V V   E     R R     0   0  ..   2
DDDD    UUU   R  RR   UUU   M   M  III        V    EEEE  R  RR    0 0   ..  22222
                                ____
                             -=(o '.
                                '.-.\\
                                /|  \\ \\
                                '|  | |
                                _\\_):)_  
"""

print(durumi_art)
headers = ["reviews", "context", "product_type", "purchase_method", "sentiment", "pros", "cons", "summary", "start_time", "end_time"]

class ChatGPTAPIResponder:
    def __init__(self):
        self.api_key = openai.api_key
        self.embedding_cache = self._preload_embeddings(["구매", "렌탈", "구독", "모름"])
        
        self.example_review_1 = "이사 갔을 때 새로 구매한 세탁기가 너무 마음에 들어요. 소음도 적고, 세탁력도 좋습니다. 다만, 가격이 조금 비싼 편이었어요."
        self.example_review_2 = "결혼 기념일에 남편이 선물해준 안마의자는 정말 최고의 선물이었어요. 매일 사용하는데, 피로가 확 풀려요."
        
        self.example_result_1 = ("이사", "세탁기", "구매", "긍정", "소음이 적고 세탁력이 좋음", "가격이 비쌈", "이사 간 집에 새로 구매한 세탁기에 대체로 만족하지만, 가격이 다소 비싸다는 점이 아쉬움.")
        self.example_result_2 = ("결혼", "안마의자", "모름", "긍정", "피로 회복에 도움", "없음", "결혼 기념일 선물로 받은 안마의자가 매일의 피로를 풀어주는데 큰 도움이 됨.")
        
        self.examples = [self.example_result_1, self.example_result_2]

    def _preload_embeddings(self, terms, model="text-embedding-ada-002"):
        embeddings = {}
        for term in terms:
            response = openai.Embedding.create(input=[term], model=model)
            embeddings[term] = response['data'][0]['embedding']
        return embeddings

    def get_embedding(self, text, model="text-embedding-ada-002"):
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']

    def get_cosine_similarity(self, emb1, emb2):
        return cosine_similarity([emb1], [emb2])[0][0]

    def get_response(self, review, engine="gpt-3.5-turbo", max_tokens=1000, temperature=0.1):
        start_time = datetime.now() 
        examples = self._get_formatted_examples()
        prompt = self._generate_prompt(review, examples)
        try:   
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "system", "content": "Analyze the following review and provide a structured analysis. Do not fabricate content that is not present. Respond accurately only to the given review. Mark non-existent content as 'none'"},
                        {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            end_time = datetime.now()  
            analysis = response['choices'][0]['message']['content'].strip().split("\n")
            analysis_result = tuple(line.split(": ")[1] for line in analysis if ": " in line)

 
            if len(analysis_result) == 7:
                best_match = self._format_analysis_result_example(*analysis_result)
                modified_analysis_result = analysis_result[:2] + (best_match,) + analysis_result[3:]
                return (*modified_analysis_result, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))

            else:
                print("there is missing elements")
                return ("none", "none", "none", "none", "none", "none", start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"), "none")
        except:
            start_idx = headers.index("context")
            end_idx = headers.index("summary") + 1
            answer_header_list = headers[start_idx:end_idx]
            ans_list = []
            for cat in answer_header_list:
                ans = self.get_simple_response(review=review,cat=cat)
                ans_list.append(ans)
            end_time = datetime.now()
            ans_list.extend([start_time,end_time])

            return tuple(ans_list)


    def get_simple_response(self, review, cat, engine="gpt-3.5-turbo", max_tokens=100, temperature=0.1):
        # 카테고리에 해당하는 인덱스 찾기
        cat_index = headers.index(cat)
        
        # 예제 프롬프트 생성
        example_prompts = []
        for example in self.examples:
            context, review_example = example[cat_index], example[0]
            example_prompts.append(f"review: {review_example}\noutput: {context}")
        
        example_prompt_str = "\n".join(example_prompts)
        
        # 시스템 프롬프트에 예제 추가
        sys_prompt = f"Given the following examples of {cat}, analyze the review:\n{example_prompt_str}\n\nNow, considering the review: {review}. \nWhat is the {cat}?"
        
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": review}],
            max_tokens=max_tokens,
            temperature=temperature)
        
        ans = response['choices'][0]['message']['content'].strip()
        
        # 구매 방법에 대한 특별 처리
        if cat == 'purchase_method':
            purchase_method_embedding = self.get_embedding(ans)
            highest_similarity = -1
            best_match = ans 
            for term, emb in self.embedding_cache.items():
                similarity = self.get_cosine_similarity(purchase_method_embedding, emb)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = term
            ans = best_match
        
        return ans
        

    def _format_analysis_result_example(self, context, product_type, purchase_method, sentiment, pros, cons, summary):
        purchase_method_embedding = self.get_embedding(purchase_method)
        highest_similarity = -1
        best_match = purchase_method 
        for term, emb in self.embedding_cache.items():
            similarity = self.get_cosine_similarity(purchase_method_embedding, emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = term
        
        return best_match  

    def _format_analysis_result(self, context, product_type, purchase_method, sentiment, pros, cons, summary):
        purchase_method_embedding = self.get_embedding(purchase_method)
        highest_similarity = -1
        best_match = purchase_method 
        for term, emb in self.embedding_cache.items():
            similarity = self.get_cosine_similarity(purchase_method_embedding, emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = term
        
        formatted_result = (
            f"1. 가전 사용 맥락: {context}\n"
            f"2. 제품 종류: {product_type}\n"
            f"3. 구매 형태: {best_match}\n"  
            f"4. 리뷰 성향: {sentiment}\n"
            f"5. 좋은점: {pros}\n"
            f"6. 아쉬운점: {cons}\n"
            f"7. 리뷰 요약: {summary}"
        )
        
        return formatted_result

    def _get_formatted_examples(self):
        formatted_example_1 = self._format_analysis_result(*self.example_result_1)
        formatted_example_2 = self._format_analysis_result(*self.example_result_2)
        
        return [
            (self.example_review_1, formatted_example_1),
            (self.example_review_2, formatted_example_2)
        ]


    def _generate_prompt(self, review, examples):
        examples_text = "\n\n".join([f"리뷰 분석:\n{ex[0]}\n\n분석 결과:\n{ex[1]}" for ex in examples])
        prompt = f"{examples_text}\n\n리뷰 분석:\n{review}\n\n분석 결과:"
        return prompt

class ReviewThread(threading.Thread):
    def __init__(self, review_queue, results_queue):
        threading.Thread.__init__(self)
        self.review_queue = review_queue
        self.results_queue = results_queue
        self.responder = ChatGPTAPIResponder()

    def run(self):
        while not self.review_queue.empty():
            review = self.review_queue.get()
            try:
                analysis_result = self.responder.get_response(review)
                self.results_queue.put((review, *analysis_result))
            finally:
                self.review_queue.task_done()

def load_reviews(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 
        return [row[0] for row in csv_reader]

def write_results(filename, results):
   
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

if __name__ == "__main__":
    if DEBUG:
        input_filename = "dataset/input_exception.csv"
        output_filename = "output/output.csv"        
    else:
        input_filename = "dataset/40.csv"
        output_filename = "output/output_40.csv"
    review_queue = Queue()
    results_queue = Queue()

    reviews = load_reviews(input_filename)
    for review in tqdm(reviews, desc="Loading reviews"):  
        review_queue.put(review)

    threads = [ReviewThread(review_queue, results_queue) for _ in range(5)]
    for thread in tqdm(threads, desc="Starting threads"): 
        thread.start()

    for thread in tqdm(threads, desc="Joining threads"):  
        thread.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    write_results(output_filename, results)