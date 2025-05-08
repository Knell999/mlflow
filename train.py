import mlflow

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# artifacts URI입니다. 실제 모델 빌드에 관련된 변동 사항들을 추적하기 위한 주소입니다.
# S3, Azure, Google Cloud Platform 등의 클라우드 저장소와의 연동도 가능합니다.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 사이킷런 모델 빌드에 관한 로그들을 기록하기 위한 코드입니다.
# autolog() 이외에 여러 메트릭을 따로 기록하는 것도 가능합니다.
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
        data_diabetes.data, data_diabetes.target
)

# Model 생성하기
reg = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=5)

# Model 훈련
reg.fit(X_train, y_train)

# Model 예측
predictions = reg.predict(X_test)

# 모델 예측 결과 출력
print(predictions)