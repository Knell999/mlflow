import mlflow
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 입출력 데이터 형식 구체화
from mlflow.models import infer_signature

# mlflow 설정
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    data_diabetes.data,
    data_diabetes.target,
    test_size=0.2,
    random_state=42,
)

# 커스텀 평가 함수 생성
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, r2, mae

if __name__ == "__main__": # 프로그램 시작점 엔트리 포인트
    
    # GridSearchCV 파라미터 설정
    #    - GridSearch에 대한 logging은 autolog()에 포함되어 있습니다.
    param_grid = {
        "n_estimators": [10, 20, 30, 40, 50, 100],
        "max_depth": [10, 15, 20],
        "max_features": [5, 6, 7, 8, 9]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # 최적의 모델 찾아서 따로 기록하기

    predictions = grid_search.best_estimator_.predict(X_train)

    # Signature를 생성하는 이유
    #    - 모델 추론시 입력 데이터의 형식을 정의하기 위함
    #    - 모델 훈련 할 때 numpy를 사용할 수도 있고, pandas를 사용할 수도 있음
    signature = infer_signature(X_train, predictions)

    # 최고의 성능을 냈었던 하이퍼 파라미터를 따로 로깅
    mlflow.log_param("n_estimators", grid_search.best_params_["n_estimators"])
    mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
    mlflow.log_param("max_features", grid_search.best_params_["max_features"])

    # 최고의 성능을 냈었던 메트릭을 따로 로깅
    rmse, r2, mae = eval_metrics(y_train, predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # 모델 저장
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        artifact_path="model",
        signature=signature,
        input_example=X_train[0:1],
        registered_model_name="diabetes_model",
    )

    