import csv
import threading
from queue import Queue
import openai
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatGPTAPIResponder:
    def __init__(self):
        self.api_key = openai.api_key

    def get_response(self, review, engine="gpt-3.5-turbo", max_tokens=250, temperature=0.5):
        start_time = datetime.now() 
        examples = self._get_formatted_examples()
        prompt = self._generate_prompt(review, examples)
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
        return (*analysis_result, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"))

    def _format_analysis_result(self, context, product_type, purchase_method, sentiment, pros, cons, summary):
        return (
            f"1. 가전 사용 맥락: {context}\n"
            f"2. 제품 종류: {product_type}\n"
            f"3. 구매 형태: {purchase_method}\n"
            f"4. 리뷰 성향: {sentiment}\n"
            f"5. 좋은점: {pros}\n"
            f"6. 아쉬운점: {cons}\n"
            f"7. 리뷰 요약: {summary}"
        )

    def _get_formatted_examples(self):
        return [
            ("이사 갔을 때 새로 구매한 세탁기가 너무 마음에 들어요. 소음도 적고, 세탁력도 좋습니다. 다만, 가격이 조금 비싼 편이었어요.",
             self._format_analysis_result("이사", "세탁기", "구매", "긍정", "소음이 적고 세탁력이 좋음", "가격이 비쌈", "이사 간 집에 새로 구매한 세탁기에 대체로 만족하지만, 가격이 다소 비싸다는 점이 아쉬움.")),
            ("결혼 기념일에 남편이 선물해준 안마의자는 정말 최고의 선물이었어요. 매일 사용하는데, 피로가 확 풀려요.",
             self._format_analysis_result("결혼", "안마의자", "선물", "긍정", "피로 회복에 도움", "없음", "결혼 기념일 선물로 받은 안마의자가 매일의 피로를 풀어주는데 큰 도움이 됨."))
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
        next(csv_reader)  # Skip header
        return [row[0] for row in csv_reader]

def write_results(filename, results):
    headers = ["reviews", "context", "product_type", "purchase_method", "sentiment", "pros", "cons", "summary", "start_time", "end_time"]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

if __name__ == "__main__":
    input_filename = "dataset/input.csv"
    output_filename = "output/output.csv"
    review_queue = Queue()
    results_queue = Queue()

    reviews = load_reviews(input_filename)
    for review in reviews:
        review_queue.put(review)

    threads = [ReviewThread(review_queue, results_queue) for _ in range(5)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    write_results(output_filename, results)
