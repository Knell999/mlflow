import mlflow
import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
        data_diabetes.data, data_diabetes.target
)

# Evaluate metrics
def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, pred)
    return mse, rmse, r2

# 성능 지표를 저장하고, 그래프로 표현하는 코드 추가
def plot_performance_metrics(mse, rmse, r2, n_estimators, max_depth):
    metrics = ['MSE', 'RMSE', 'R2']
    values = [mse, rmse, r2]
    
    fig = plt.figure(figsize=(8, 4))
    plt.title(f"n_estimators : {n_estimators}, max_depth : {max_depth}")
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.xlabel('Performance Metrics')
    plt.ylabel('Values')
    plt.ylim(0, max(values) + max(values) * 0.1)
    plt.savefig(f"RFR_n_estimators_{n_estimators}_max_depth_{max_depth}.png")
    plt.close(fig)
    
if __name__ == "__main__":

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else None
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    with mlflow.start_run():
        # autolog()는 fit()이 호출되었을 때 사용됩니다. 파라미터와 메트릭만 기록하고 끝납니다.
        # 따라서 start_run()을 이용해 컨텍스트 내에서 autolog()를 호출하면 따로 기록하고자 하는 내용 까지도 기록됩니다.
        mlflow.sklearn.autolog()

        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        reg.fit(X_train, y_train)
        
        mse, rmse, r2 = eval_metrics(y_test, reg.predict(X_test))
        plot_performance_metrics(mse, rmse, r2, n_estimators, max_depth)
        
        mlflow.log_artifact(f"RFR_n_estimators_{n_estimators}_max_depth_{max_depth}.png")