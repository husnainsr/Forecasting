from flask import Flask, render_template, jsonify, request
import sqlite3

app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/results')
def results_page():
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation")
    evaluations = cursor.fetchall()
    conn.close()
    return render_template('results.html', evaluations=evaluations)

@app.route('/comparison')
def comparison_page():
    return render_template('comparison.html')

@app.route('/get_model_names')
def get_model_names():
    conn = sqlite3.connect('your_database.db')  
    cursor = conn.cursor()
    cursor.execute("SELECT model_name FROM evaluation")
    model_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(model_names)

@app.route('/get_evaluation_data')
def get_evaluation_data():
    model_name = request.args.get('model')
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT mse, r_square, rmse, mae, mape FROM evaluation WHERE model_name=?", (model_name,))
    data = cursor.fetchone()
    conn.close()
    return jsonify({
        'mse': data[0],
        'r_square': data[1],
        'rmse': data[2],
        'mae': data[3],
        'mape': data[4]
    })

@app.route('/get_actual_predicted_data')
def get_actual_predicted_data():
    model_name = request.args.get('model')
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT index_value, Test, {} FROM testing".format(model_name))
    data = cursor.fetchall()
    labels = [row[1] for row in data]
    index_values = [row[0] for row in data]  # Generate index values
    values = [row[2] for row in data]
    conn.close()
    return jsonify({'labels': labels, 'index_values': index_values, 'values': values})

if __name__ == '__main__':
    app.run(debug=True)
