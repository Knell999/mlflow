import mlflow
import numpy as np
import sys

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
        data_diabetes.data, data_diabetes.target
)

if __name__ == "__main__":
    # RandomForest의 하이퍼 파라미터를 명령줄에서 입력
    #    명령줄로 입력되는 모든 정보는 문자열 형식으로 들어오기 때문에, 파싱이 필요할 수 있다.
    #    sys.argv[1] 부터 사용한다.
    #        아무것도 넣지 않으면 sys.argv에 [train_rf_sys_argv.py] 이렇게 들어온다.
    #        즉 기본 길이(len)는 1이다.
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    reg.fit(X_train, y_train)