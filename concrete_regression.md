### 📌 콘크리트 강도 회귀 예측

#### 📚 Features

-   cement: 시멘트(kg/m<sup>3</sup>)
-   slag: 광재(kg/m<sup>3</sup>)
-   ash: 비산회(시멘트 혼합제, kg/m<sup>3</sup>)
-   water: 물(kg/m<sup>3</sup>)
-   superplastic: 유동화제(kg/m<sup>3</sup>)
-   coarseagg: 굵은 골재(kg/m<sup>3</sup>)
-   fineagg: 잔골재(kg/m<sup>3</sup>)
-   age: 콘크리트가 형성된 후 경과한 시간(1~365)

#### 🎈 Target

-   strength: 콘크리트의 압축강도

---

#### 📌 목차

1. 데이터 탐색 및 전처리
2. 선형 회귀 분석
3. 다항 회귀 분석
4. 트리 기반 회귀 모델을 사용한 회귀 분석
5. OLS, VIF 확인
6. 과적합 확인을 위한 교차 검증 및 예측값/실제 정답 비교

#### 1. 데이터 탐색 및 전처리

-   결측치가 존재하지 않았으며, 약 25개의 중복행을 삭제하였습니다.
-   독립변수와 종속변수를 포함하여 상관관계를 히트맵으로 시각화한 결과는 아래와 같았습니다.

<img src="./images/concrete_corr_heatmap.png">

-   독립변수 사이의 상관관계가 0.5 이상인 독립변수는 존재하지 않았습니다.
-   종속변수와의 상관관계가 어느 정도 나타나는 cement, superplastic, age 등의 상관관계 수치를 확인해봅니다.
-   종속변수와의 상관관계를 내림차순으로 출력한 결과는 아래와 같습니다.

          cement          0.488283
          superplastic    0.344209
          age             0.337367
          slag            0.103374
          ash            -0.080648
          coarseagg      -0.144717
          fineagg        -0.186448
          water          -0.269624

#### 2. 선형 회귀 분석

-   기초적인 전처리만 진행한 상태에서, scikit-learn의 LinearRegression 모델을 통해 선형 회귀 예측을 진행한 결과 평가 지표 점수는 아래와 같았습니다.
-   `MSE: 100.9549, RMSE: 10.0476, MSLE: 0.1098, RMSLE: 0.3314, R2: 0.5974`
-   R<sup>2</sup>가 약 0.5974로, 해당 수치만으로 데이터의 선형성/비선형성을 판단하기엔 무리가 있지만,  
    다항 회귀 예측 및 트리 기반 회귀 모델을 사용한 비선형 회귀 예측을 진행하여  
    R<sup>2</sup>를 비교해보고자 했습니다.

#### 3. 다항(Polynomial) 회귀 분석

-   다항 회귀의 경우, `PolynomialFeatures()`의 하이퍼파라미터인 degree를 2차로 설정했을 때  
    R<sup>2</sup>가 약 0.7714로 가장 높았습니다(degree=3일 경우 약 0.7640으로 감소).
-   `MSE: 57.3214, RMSE: 7.5711, MSLE: 0.0582, RMSLE: 0.2413, R2: 0.7714`
-   따라서 선형 회귀 모델인 LinearRegression 모델을 사용하면서도,  
    비선형 관계를 모델링하는 방법인 다항 회귀를 사용하여 변수들을 다항식으로 변환한 후  
    사용함으로써 비선형성을 적용하였을 때 R<sup>2</sup>가 약 0.2 가량 유의미하게 상승하였으므로,  
    해당 데이터를 비선형 데이터로 간주하는 것에 무리가 없음을 확인하였습니다.

#### 4. 트리 기반 회귀 모델을 사용한 회귀 분석

-   해당 데이터의 비선형성을 앞서 확인하였으므로, 트리 기반 모델을 사용하여 회귀 분석을 진행합니다.
-   사용한 트리 기반 모델은 아래와 같습니다:

    1. DecisionTreeRegressor
    2. RandomForestRegressor
    3. GradientBoostingRegressor
    4. XGBRegressor
    5. LGBMRegressor

-   각 모델별 RMSE(루트 평균 제곱 오차)와 R<sup>2</sup>는 아래와 같습니다.

    1. DecisionTreeRegressor - RMSE: 6.0435, R2: 0.8544
    2. RandomForestRegressor - RMSE: 5.0128, R2: 0.8998
    3. GradientBoostingRegressor - RMSE: 5.3812, R2: 0.8845
    4. XGBRegressor - RMSE: 4.2289, R2: 0.9287
    5. LGBMRegressor - RMSE: 4.4983, R2: 0.9193

<img src="./images/concrete_evaluation_by_models.png">

-   이 중 XGBRegressor(XGBoost) 모델이 RMSE와 R<sup>2</sup> 기준 가장 좋은 성능을 보였으므로,  
    해당 모델을 사용하여 GridSearchCV를 통해 최적의 하이퍼파라미터를 찾고자 했습니다.
-   각 하이퍼파라미터 조합에 대해 KFold를 통해 무작위로 10번의 교차 검증을 수행한 결과는 아래와 같았습니다.

<img src="./images/concrete_gridsearchcv.png" width="400px">

-   GridSearchCV의 `cv_results_`를 통해 확인해본 결과,  
    `max_depth`가 작을 수록, 그리고 대체로 `n_estimators=100`일 때 R<sup>2</sup>가 높았습니다.
-   `best_estimator_`로 테스트 데이터에 대한 회귀 예측을 진행한 결과는 아래와 같았습니다.
-   `MSE: 17.3164, RMSE: 4.1613, MSLE: 0.0232, RMSLE: 0.1523, R2: 0.9309`
-   과적합 여부 등을 판단하기 위해 OLS 및 VIF를 확인해봅니다.

#### 5. OLS 및 VIF 확인

-   OLS 결과, R<sup>2</sup>는 약 0.930, Durbin-Watson은 약 1.862로 문제 없이 나타났습니다.
-   또한 각 독립변수의 p-value는 모두 0.003 이하로 나타났습니다.
-   VIF를 통해 다중공선성을 확인한 결과는 아래와 같았습니다.

<img src="./images/concrete_vif_before.png" width="200px">

-   water, coarseagg, fineagg의 VIF가 높아 종속변수와의 상관관계를 확인해본 결과,
    각각 약 -0.26, -0.14, -0.18로 낮게 나타나 삭제한 후 다시 OLS와 VIF를 확인하였습니다.
-   OLS 상에서 R<sup>2</sup>는 약 0.920, Durbin-Watson은 약 1.864로 나타났으며, VIF는 아래와 같았습니다.

<img src="./images/concrete_vif_after.png" width="200px">

-   모든 feature의 VIF 수치가 3 미만으로, 다중공선성 문제가 해결되었습니다.
-   다시 XGBRegressor 모델로 회귀 예측을 수행한 결과는 아래와 같았습니다.
-   `MSE: 18.8915, RMSE: 4.3464, MSLE: 0.0214, RMSLE: 0.1462, R2: 0.9247`
-   3개의 feature를 제거하기 전의 R<sup>2</sup>는 약 0.9309, 제거 후에는 약 0.9247로 나타났으며,  
    제거 전의 RMSE는 약 4.1613, 제거 후에는 약 4.3464로 나타났습니다.
-   높은 다중공선성을 보이던 3개의 feature를 제거한 후 R<sup>2</sup>는 약 0.006 감소하였지만,  
    여전히 높은 수치를 보이고 있다는 점에서 오히려 더 높은 신뢰도를 가진 회귀 모델로 판단됩니다.

#### 6. 과적합 확인을 위한 교차 검증 및 예측값/실제 정답 비교

-   `cross_val_score()`을 통한 교차 검증 결과 평균 R<sup>2</sup>는 약 0.8924,  
    실제 예측 결과 R<sup>2</sup>는 약 0.9247로 교차 검증 시에는 큰 문제가 발견되지 않았습니다.
-   학습 데이터와 테스트 데이터에 대한 모델의 예측값과 실제 정답의 분포를 시각화한 결과는 아래와 같습니다.

<img src="./images/concrete_prediction_label_best.png">

<code>MSE: 1.8634, RMSE: 1.3651, MSLE: 0.0022, RMSLE: 0.0469, R2: 0.9931(학습 데이터)  
MSE: 18.8915, RMSE: 4.3464, MSLE: 0.0214, RMSLE: 0.1462, R2: 0.9247(테스트 데이터)</code>

-   테스트 데이터의 R<sup>2</sup>가 낮으므로, 과적합 발생을 의심해볼 수 있었습니다.
-   따라서 `best_estimator_`가 아닌, GridSearchCV에서 R<sup>2</sup> 기준 12위를 기록한 하이퍼파라미터  
    조합으로 모델을 생성하여 학습 및 예측을 수행한 후, 같은 방식으로 시각화하였습니다.

<img src="./images/concrete_prediction_label_bad.png">

<code>MSE: 1.6297, RMSE: 1.2766, MSLE: 0.0018, RMSLE: 0.0424, R2: 0.9939(학습 데이터)  
MSE: 30.3917, RMSE: 5.5129, MSLE: 0.0365, RMSLE: 0.1911, R2: 0.8788(테스트 데이터)</code>

-   테스트 데이터의 R<sup>2</sup>가 더 낮아짐에 따라, 이전의 모델에서 과적합이 발생했다고 보기 어렵다는  
    결론을 이끌어낼 수 있었습니다.
-   따라서 `XGBRegressor(max_depth=4, n_estimators=500, random_state=124) 모델이 가장 적합한  
    성능을 보였다고 볼 수 있습니다.
