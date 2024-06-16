from fastapi import APIRouter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fastapi.responses import JSONResponse

from train_test_evaluation.controller.response_form.analysis_result_response_form import AnalysisResultResponseForm

trainTestEvaluationRouter = APIRouter()


@trainTestEvaluationRouter.get("/train-test-evaluation", response_model=AnalysisResultResponseForm)
async def analyze():
    # 1. 데이터 로드 및 준비
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. 모델 훈련
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 3. 모델 테스트 및 평가
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)

    selected_metrics = []

    for key in class_report.keys():
        if key in ['setosa', 'versicolor', 'macro avg', 'weighted avg']:
            selected_metrics.append({
                "metric": key,
                "precision": class_report[key]['precision'],
                "recall": class_report[key]['recall'],
                "f1-score": class_report[key]['f1-score']
            })

    return JSONResponse({
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": selected_metrics
    })
