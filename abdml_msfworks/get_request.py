from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/scada-optimization/retrain", methods=["POST"])
def retrain():
    data = request.get_json(force=True)

    print("\nRECEIVED RETRAIN REQUEST:")
    print(data)

    # Later: trigger your retraining logic here
    # start_retrain_pipeline(data)

    return jsonify({
        "status": "ok",
        "message": "retrain request received"
    })

@app.route("/health", methods=["GET"])
def health():
    return "OK"

if __name__ == "__main__":
    print("Listening on http://0.0.0.0:6196")
    app.run(host="0.0.0.0", port=6196)



"""from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/receive", methods=["POST"])
def receive():
    data = request.get_json(force=True)
    print("\n RECEIVED JSON:")
    print(data)

    return jsonify({
        "status": "ok",
        "received": True
    })

@app.route("/health", methods=["GET"])
def health():
    return "OK"

if __name__ == "__main__":
    print(" Listening on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
"""