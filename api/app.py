from flask import Flask, request, jsonify
import middleware
import constants


app = Flask(__name__)

@app.route('/recommendation', methods=['GET'])
def recommendation():
    customer_id = request.args.get(constants.CUSTOMER_ID)
    top = request.args.get(constants.TOP)
    recommendations = None

    if not customer_id:
        return jsonify({"error": "Debe proporcionar un customer_id"}), 400
    else:
        recommendations = middleware.generateRecommendation(user=customer_id, top=top)
    
    return jsonify({
        "customer_id": customer_id,
        "recommendations": recommendations
    })

@app.route("/saludo/<nombre>", methods=["GET"])
def saludo(nombre):
    return {"message": f"Hola, {nombre}!"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
