#regression 
curl -X POST http://localhost:8000/predict/regression -H "Content-Type: application/json" -d '{"features": [1, 2, 3, 4, 5,6]}'

#xgboost
curl -X POST http://127.0.0.1:8000/predict/xgboost \
     -H "Content-Type: application/json" \
     -d '{"features": [3, 22.0, 7.25]}'