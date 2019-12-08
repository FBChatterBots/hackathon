from flask import Flask, jsonify, request
from PIL import Image
from io import BytesIO
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  if (request.method == 'POST'):
    some_json = request.get_json()
    print(some_json)
    return jsonify({'you sent this': some_json}), 201
  else:
    return jsonify({"about" : "Hello World!"})
  
# @app.route('/multi/<int:num>', methods={'GET'})
# def get_multiply10(num):
#   return jsonify({'result' : num*10})

@app.route('/text/<string:stuff>', methods={'GET'})
def run_file(stuff):
  
  return jsonify({'result is' : stuff})


@app.route('/multiply/<int:num>', methods={'GET'})
def multiply(num):
  
  return jsonify({'Multiplied by 10' : num*10})

if __name__ == '__main__':
    app.run(debug=True)