from flask import Flask, render_template, request
import Iris_model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sepal_length = request.form['sepalLen']
        sepal_width = request.form['sepalWid']
        petal_length = request.form['petalLen']
        petal_width = request.form['petalWid']

        new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        model = Iris_model.model()
        y_pred = model.predict(new_data)

        if y_pred == 0:
            return render_template('index.html', setosa='Setosa')
        elif y_pred == 1:
            return render_template('index.html', versicolor='Versicolor')
        else:
            return render_template('index.html', virginica='Virginica')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)
