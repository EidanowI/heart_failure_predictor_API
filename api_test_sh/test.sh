#!/bin/bash

CSV_FILE="../dataset/heart_test.csv"  # Укажите путь к вашему CSV файлу

# Пропускаем заголовок и обрабатываем каждую строку
tail -n +2 "$CSV_FILE" | while IFS=',' read -r Age Sex ChestPainType RestingBP Cholesterol FastingBS RestingECG MaxHR ExerciseAngina Oldpeak ST_Slope HeartDisease; do
    
    # Формируем JSON запрос
    JSON_DATA=$(cat <<EOF
{
    "Age": $Age,
    "Sex": "$Sex",
    "ChestPainType": "$ChestPainType",
    "RestingBP": $RestingBP,
    "Cholesterol": $Cholesterol,
    "FastingBS": $FastingBS,
    "RestingECG": "$RestingECG",
    "MaxHR": $MaxHR,
    "ExerciseAngina": "$ExerciseAngina",
    "Oldpeak": $Oldpeak,
    "ST_Slope": "$ST_Slope"
}
EOF
)
    RESPONSE=$(curl -s -X POST "http://127.0.0.1:8000/perceptron/predict" \
        -H "Content-Type: application/json" \
        -d "$JSON_DATA")
    
    echo "$RESPONSE"
    echo "---(real is heart disease: $HeartDisease)"
    
done

echo "Обработка завершена"