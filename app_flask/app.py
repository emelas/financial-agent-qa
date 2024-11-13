
from utils import qa_advanced_scorer,qa_advanced_scorer_v1,qa_basic_scorer
from flask import Flask, request, jsonify
app = Flask(__name__)

def make_serialisable(obj):
    if isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {key: make_serialisable(value) for key, value in obj.items()}
        else:  # obj is a list
            return [make_serialisable(item) for item in obj]
    else:
        return str(obj)  # Convert any non-dict and non-list type to string

@app.route('/process_document', methods=['POST'])
def process_document():

    data = request.get_json()

    print(f'data: {data}')

    text = data.get('text')
    question = data.get('question')
    ground_truth = data.get('ground_truth')

    result = qa_basic_scorer.qa_maths_reasoning_langgraph(text, question, ground_truth)

    serialisable_result = make_serialisable(result)

    # return {'response':result}
    return jsonify({'response': serialisable_result})

@app.route('/process_document_advanced', methods=['POST'])
def process_document_advanced():

    data = request.get_json()

    print(f'data: {data}')

    text = data.get('text')
    question = data.get('question')
    ground_truth = data.get('ground_truth')

    result = qa_advanced_scorer_v1.qa_maths_reasoning_langgraph_advanced_scorer(text, question, ground_truth)

    serialisable_result = make_serialisable(result)

    # return {'response':result}
    return jsonify({'response': serialisable_result})

@app.route('/process_document_advanced_agent', methods=['POST'])
def process_document_advanced_agent():

    data = request.get_json()

    print(f'data: {data}')

    text = data.get('text')
    question = data.get('question')
    ground_truth = data.get('ground_truth')

    result = qa_advanced_scorer.qa_maths_reasoning_langgraph_advanced_scorer(text, question, ground_truth)

    serialisable_result = make_serialisable(result)

    # return {'response':result}
    return jsonify({'response': serialisable_result})


if __name__ == '__main__':
    app.run(debug=True)