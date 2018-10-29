from evaluate import load_model, evaluate
import dill as pickle
from flask import Flask
from flask_restful import reqparse, Api, Resource
import torch
from model.utils import AppData

app = Flask(__name__)
api = Api(app)

model_args = pickle.load(open('./pickle/app_args.pkl', 'rb'))
model_args.device = torch.device('cuda:0' if torch.cuda.
                                 is_available() else 'cpu')

parser = reqparse.RequestParser()
parser.add_argument('q1')
parser.add_argument('q2')


class PredictSentenceSimilarity(Resource):
    def get(self):
        app_args = parser.parse_args()
        q1, q2 = app_args['q1'], app_args['q2']

        model_data = AppData(model_args, [q1, q2])
        model = load_model(model_args, model_data)
        preds = evaluate(model, model_args, model_data, mode='app').numpy()

        return {'preds': preds}


api.add_resource(PredictSentenceSimilarity, '/')

if __name__ == '__main__':
    app.run(debug=True)
