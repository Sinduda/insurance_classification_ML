# 랜덤포레스트 모델을 이용해 자동차 보험 가입에 관심있는 고객을 예측하는 보조 모델
### (코드스테이츠 머신러닝 섹션 개인 프로젝트)
- 진행 기간: 5월 19일 ~ 5월 24일
- 사용 언어: `Python`
- 프로젝트 발표 영상은 ([여기](https://youtu.be/53A-bCQj3sc))에서 보실 수 있습니다. (청자는 **비데이터 직군**으로 가정)

---

<br>

## **프로젝트 개요**

랜덤포레스트 모델을 이용해 **자동차 보험 가입에 관심있는 고객을 예측하는 보조 모델**

> 👉🏻  건강보험 가입 고객 중 새로 출시되는 자동차 보험에도 관심있는 고객을 예측하여, 출시 초기 효율적인 Lead 확보를 도와주는 머신러닝 보조 모델 개발.
> 
> - 보험 회사에서 기존 건강보험 가입자를 대상으로 자동차 보험에도 가입할 의사가 있는지를 예측하는 모델을 만드는 것이 프로젝트의 목표입니다. (`0,1 Classification`)
> 1. [데이터셋](https://www.kaggle.com/datasets/pawan2905/jantahack-cross-sell-prediction?select=train.csv)을 `matplotlib`, `seaborn`을 이용해 다양하게 시각화하며, 건강 보험 가입 고객 중 자동차 보험 가입에도 관심있다고 한 고객들의 특징을 확인하였습니다.
> 2. 평가 지표는 타겟 클래스 분포가 불균형하여 `f1_score`를, 기준 모델은 랜덤포레스트를 선택하였습니다.
> 3. 랜덤포레스트(Random/Grid Search CV), XGBoost 등의 모델 실험 및 비교 후, **최종적으로 데이터를 DownSampling한 후 RandomSearch CV를 통해 하이퍼 파라미터를 최적화한 랜덤포레스트 모델을 채택했습니다.**
> 4. `Permutation Importance`, `PDP(Partial Dependence Plot)`을 통해 타겟값에 가장 많은 영향을 주는 특성을 확인하여 모델을 해석했습니다.
> 
> ---
>

<br/>

## 문제 **정의 및 사용 데이터셋**

**[시나리오 가정]**

- 건강 보험을 성공적으로 운영하고 있는 우리 보험사는 **이제 새로운 자동차 보험의 출시를 앞두고 있다.**
- 새로운 상품을 출시할 때 가장 중요한 것은, **실제로 해당 상품에 관심 있는 고객에게 더 집중하여 한정된 자원을 효율적으로 사용하는 것**이다.
- 따라서 **기존 건강 보험 가입 고객 중 자동차 보험에도 관심 있을 고객을 예측하는 것이 필요한 상황이다.**

**[문제 정의]**

- **이미 건강 보험에 가입해 있는 고객 중 자동차 보험 가입에도 관심 있는 고객을 예측하는 ML 분류 모델을 만들어, 출시 초기 효율적으로 영업 리소스를 분배하는데 활용할 수 있도록 도와주자.**

**[사용한 데이터셋]**

- 캐글에 있는 [JantaHack: Cross sell Prediction](https://www.kaggle.com/datasets/pawan2905/jantahack-cross-sell-prediction?select=train.csv) 데이터셋을 활용하였습니다.
- 원본 데이터의 형태 및 주요 구성은 아래와 같습니다.
    <img width="1150" alt="dataset" src="https://user-images.githubusercontent.com/81467922/184593985-fb84cb1a-88ee-4741-9be0-6852f0bfc372.png">
    
    > `성별`, `나이`, `면허 보유 여부`, `지역 코드`, `자동차 보험 가입 여부`, `자동차 연식`, `사고 이력`, `보험료`, `소통 채널`, `고객과 함께한 기간` <br>
    > **Target** - `Response`(자동차 보험에 가입할 의사가 있는지 응답)
    > - `shape = (381109,12)`
    
<br/>

**[간단한 탐색]**

- `Pandas Profiling`을 통해 데이터를 간단히 탐색하고 아래와 같은 가설을 정리하였습니다.
    
    ```markdown
    - 운전면허가 없는 고객의 대다수는 자동차 보험에 관심이 없을 것이다.
    - 나이가 어릴수록 자동차 보험 가입에 관심이 없을 것이다.
        - 보험료를 감당할 수 있는가? 운전에 대한 근거 없는 자신감?
    - 연식이 오래되었을 수록 관심이 없을 것이다.
        - 이게 맞다면 이런 질문도 의미O ⇒ 나이가 어릴 수록 자동차 연식이 오래된 걸 탈 것이다.
    - 성별과 관심도의 차이는 없을 것이다.
    - 이미 자동차 보험에 가입한 고객은 우리 회사의 자동차 보험 가입에 관심이 없을 것이다.
    - 이전에 자동차 파손이 있었던 고객은 보험 가입에 관심이 있을 것이다.
    - 연간 보험료가 낮을수록 가입에 관심이 많을 것이다.
    - 세일즈 채널과 관심도의 차이는 없을 것이다.
    - 고객이 우리 회사와 오래 했을 수록 관심이 많을 것이다.
    ```
    

**[데이터 전처리]**

- 결측치 및 데이터 타입, 이상치를 확인하였고 큰 특이사항이 없어 데이터 전처리는 진행할 필요가 없었습니다.
- 학습 시점에 알 수 없는 정보가 포함되었는지 `타겟 누수(Target Leakage)`를 체크하였고, 이상이 없음을 확인하였습니다.

**[EDA 및 데이터 시각화]**

- 분류 모델을 만들 것이기 때문에 가장 먼저 타겟(`Response`) 클래스 분포를 시각화했고, 모델 학습 시 `class_weight`에 유의해야 한다는 점을 확인했습니다.
    - 1: 관심있음 => 0.88, 0: 관심없음 => 0.12
    
- 판다스 프로파일링을 통해 세운 가설들을 `barplot`, `violinplot` 등을 통해 데이터를 시각화하여 인사이트를 도출하였습니다.
- **Main Insight**
    - **1)사고이력 있고, 2)이미 가입된 자동차 보험이 없는 사람을 타겟으로 하는 것이 좋을 것 같다.**
    
 <br/>
 
## 모델링

**[기준 모델]**

- 타겟 클래스 분포가 불균형했기 때문에 평가 지표로 `accuracy`를 쓰기 보단 `f1_score`를 선택했고, 따라서 **디폴트 설정의 랜덤 포레스트 모델을 기준 모델로 선정하였습니다.**
    - 기준 모델 성능
    
      <img width="408" alt="baseline" src="https://user-images.githubusercontent.com/81467922/184594596-2c827ce5-fa51-48c1-8958-4c239db6d6cf.png">
  

**[모델링 - 전략 설정]**

- 타겟 클래스 불균형 문제로 인한 기준 모델 선정에 시간이 지체되어, 주어진 기간 내 프로젝트를 마무리하기에 시간이 촉박한 상황이었습니다.
- 따라서 **“가능한 기본 형태로 빠르게 여러 개의 모델을 돌려보고, 가장 성능이 나은 모델을 선택해 고도화해보자"는 전략**으로 본격적인 모델링을 진행했습니다.

**[모델링 - 진행 과정]**

1. **랜덤 포레스트 - RandomSearchCV**
    - 테스트셋 성능
    
        <img width="408" alt="rf_cv" src="https://user-images.githubusercontent.com/81467922/184594688-5457eb91-36b1-45e6-bbf6-e4e3e739cf93.png">        
    
2. **랜덤 포레스트 - GridSearchCV**
    - 테스트셋 성능
    
        <img width="408" alt="rf_grid" src="https://user-images.githubusercontent.com/81467922/184594771-23572a7a-7348-4ec7-b75f-b6226be85921.png">

    
3. **XGBoost**
    - 테스트셋 성능
    
        <img width="408" alt="xgboost" src="https://user-images.githubusercontent.com/81467922/184594840-a8c7ce7d-6455-452a-b8e8-178a09178771.png">
        
- **결론: `f1_score` 및 `recall`이 우수한 1, 3번 모델을 가지고 본격적으로 모델링을 진행해보자.**
4. **위 과정을 훈련 데이터를 다운 샘플링한 이후 다시 반복해 실행하였습니다.**
    1. DownSampling.
        - `sklearn`의 `resample` 모듈을 이용해 `labal`의 비율은 5:5로 맞추어 다운샘플링.
        - `X_train.Shape = (70064,11)`
    2. 다운샘플링한 데이터를 이용하여 위 1, 3번 모델에 재학습하고 성능 확인.

**최종적으로는 RandomSearchCV를 통해 하이퍼 파라미터를 최적화한 아래 모델을 채택하였습니다.**

    pipe = make_pipeline(RandomForestClassifier(random_state=42))

    dists = {
    'randomforestclassifier__n_estimators': randint(50, 1000),
    'randomforestclassifier__max_depth': [10, 20, 50, None],
    'randomforestclassifier__min_samples_leaf': randint(1,20),
    'randomforestclassifier__max_features': uniform(0, 1) # max_features
    }

    clf = RandomizedSearchCV(
    pipe,
    param_distributions=dists,
    n_iter=10,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
    )

    clf.fit(X_train_downsample, y_train_downsample);

- 모델 성능

    <img width="408" alt="score" src="https://user-images.githubusercontent.com/81467922/184594933-8d122aa1-a61a-4c37-baba-39de8daaf26f.png">
    
    - `precision`과 `recall` 중, `recall`에 더 중요성을 두었습니다.
        - 진짜 원하는 고객을 놓치지 않고 운영 리소스를 배분하는 것이 더 중요하다고 생각했기 때문.
        
<br>

## **모델 해석**

**[Permutation Importances]**

- 최종적으로 선택한 모델에서 타겟에 가장 많은 영향을 주는 특성을 확인하기 위해 특성 중요도를 확인하였습니다. *(`eli5` 라이브러리의 `PermutationImportance` 모듈 활용)*
    
    <img width="408" alt="pi" src="https://user-images.githubusercontent.com/81467922/184595265-18790deb-b7d4-46fa-9055-ba0457df5f2b.png">    
    
    **⇒ 과거 사고이력 여부와 이미 가입된 자동차 보험이 있는지 여부가 가장 중요한 특성으로 판단됨.**
    

**[PDP]**

- 앞서 중요한 특성으로 확인된 두 가지 특성에 대해 타겟과의 관계를 PDP를 그려 확인하였습니다.

    <img width="662" alt="pdp" src="https://user-images.githubusercontent.com/81467922/184595320-8aea9763-448f-40ea-b539-f6b6ecefaeac.png">
    
    ⇒ 초반에 설정한 가설인 “1)사고이력 있고, 2)이미 가입된 자동차 보험이 없는 사람을 타겟으로 하는 것이 좋을 것 같다.”가 제가 만든 모델에서 맞을 확률이 높다는 점을 확인할 수 있었습니다.
    
<br>

## **아쉬운 점**

- 시간 관계상 좀 더 많은 특성 공학과 다양한 모델링을 해보지 못한 점이 아쉬웠습니다.
- 프로젝트 발표 간 모델 성능 개선에 대한 향후 방향성 제시가 부재하는데, 이에 대한 제시도 함께 이루어졌다면 더 좋았을 것 같습니다.

<br>

## **프로젝트 회고**

- 이번 프로젝트를 완료한 직후 작성한 회고 글은 아래에서 읽어보실 수 있습니다.
    
    [220519-0524 머신러닝 개인 프로젝트 회고](https://velog.io/@cualquier/220519-0524%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EA%B0%9C%EC%9D%B8-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%9A%8C%EA%B3%A0#3-%EB%AA%A8%EB%8D%B8%EB%A7%81-%EB%B0%8F-%EB%AA%A8%EB%8D%B8-%ED%95%B4%EC%84%9D)
