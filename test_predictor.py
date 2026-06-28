from utils.predictor import predict_patient

sample = {

    "age":30,

    "gender":"Male",

    "hemoglobin":13.5,

    "wbc":7000,

    "differential":60,

    "rbc":5,

    "platelet":180000,

    "pdw":15

}

result = predict_patient(sample)

print(result)