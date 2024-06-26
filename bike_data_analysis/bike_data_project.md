### 🎈 ZMG 지역의 성별에 따른 자전거 대여 서비스 이용률 차이가 나타난 원인은 무엇인가?

#### 📌분석 대상 데이터

##### 1) 2014.12.~2024.01 기간, ZMG(Zona Metropolitana de Guadalajara) 지역에서의 MiBici 사 자전거 대여 서비스 이용 데이터

##### 2) ZMG 지역에서의 MiBici 자전거 대여소 데이터

##### 3) 할리스코(Jalisco) 주의 인구 피라미드 데이터

##### 4) 할리스코(Jalisco) 주의 성별에 따른 노동 인구 데이터

##### 5) 할리스코(Jalisco) 주의 성별에 따른 통근 시간 데이터

##### 6) 할리스코(Jalisco) 주의 연령대 및 성별에 따른 세대주 데이터

##### 7) 할리스코(Jalisco) 주의 통근 시간에 따른 통근 수단 이용 데이터

#### **✏️주제: ZMG 지역의 성별에 따른 자전거 대여 서비스 이용률 차이가 나타난 원인은 무엇인가?**

---

#### **📌목차**

1. 데이터 전처리

2. 데이터 시각화 및 분석

3. 결론

#### 1. 데이터 전처리

##### 1) 자전거 대여 서비스 이용 데이터

###### 📚Columns

-   Trip_Id: 고유 id
-   User_Id: 사용자 id
-   Sex: 사용자의 성별(M: 남성, F: 여성)
-   Birth_year: 사용자의 출생연도
-   Trip_start: 대여 서비스 시작 날짜 및 시간
-   Trip_end: 대여 서비스 종료(반납) 날짜 및 시간
-   Origin_Id: 대여한 장소의 대여소 id
-   Destination_Id: 반납한 장소의 대여소 id
-   Age: 사용자의 나이(2024년 기준, 만 나이 기준)
-   Duration: 대여 기간(Trip_end - Trip_start)

<img src="../images/bike_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   인덱스를 나타내는 'Unnamed: 0' 컬럼을 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   모델 학습 목적이 아닌 분석 목적이므로 이상치에 민감하지 않을 수 있지만,  
    정상 범위에서 크게 벗어난 데이터가 있는지 검사해보았습니다.
-   출생연도(Birth_year) 컬럼과 출생 연도를 바탕으로 한 나이(Age) 컬럼의 분포를 `describe()`를 통해 확인해 보았습니다.

<img src="../images/bike_birth_year_describe.png">
<img src="../images/bike_age_describe.png">

-   2024년 현재 기준으로 100세 이상인 1924년 이전 출생 데이터를 대상으로 대여 연도를 확인해본 결과 아래와 같았습니다.

<img src="../images/bike_over_100_trip_starts.png" width="200px">

-   나이가 91세 이상인 데이터를 대상으로 출생연도와 나이를 대체하기로 판단하였습니다.
-   해당 데이터를 제외한 데이터의 출생연도 평균값으로 출생연도를 대체하고,  
    대체한 출생연도를 바탕으로 2024년 현재의 나이를 구하여 나이 데이터를 대체하였습니다.
-   출생연도의 평균값은 약 1987.23 이었으며, 따라서 출생연도는 1987, 나이는 37로 대체하였습니다.

##### 2) 자전거 대여소 데이터

###### 📚Columns

-   id: 대여소 id
-   name: 대여소의 이름(스페인어)
-   obcn: 대여소 code
-   location: 대여소의 위치
-   latitude: 대여소의 위도
-   longitude: 대여소의 경도
-   status: 대여소의 상태

<img src="../images/bike_station_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   obcn(대여소 고유 코드)는 이미 고유 ID인 id 컬럼이 존재하며, 앞선 데이터와 id를 기준으로 join할 수 있으므로 삭제하였습니다.
-   또한 위도(latitude)와 경도(longitude)로 지도 상의 위치를 알 수 있으므로 location 컬럼을 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   위도와 경도의 4분위 분포도를 확인한 후, ZMG 지역 안에서 유효한 위치를 나타내고 있는지 검사하였습니다.

<img src="../images/bike_station_lat_long_describe.png">

-   각 데이터의 위치를 위도와 경도 정보를 이용하여 `folium` 라이브러리를 통해 마커로 표시하였습니다.

<img src="../images/bike_station_folium_map.png">

-   지도에서 확인한 결과 위도/경도에 따른 대여소의 위치가 ZMG 구역 내에 정상적으로 위치함을 확인할 수 있습니다.

###### (4) name 컬럼의 값 수정

-   name 컬럼의 값 앞에 이전에 삭제한 obcn 코드의 값이 포함되어 있으므로, 해당 부분을 삭제하였습니다.
-   삭제 후의 데이터는 아래와 같습니다.

<img src="../images/bike_station_preprocessed_data.png">

##### 3) 할리스코(Jalisco) 주의 인구 피라미드 데이터

###### 📚Columns

-   Sex ID: 성별 id
-   Sex: 성별
-   Age Range ID: 연령대(5년 단위) id
-   Age Range: 연령대(5년 단위)
-   Population: 인구수
-   Population\_: 인구수(음수)
-   Percentage: 전체 인구 대비 해당 연령대/성별의 인구 비율

<img src="../images/bike_pyramid_original_data.png">

<sub>`head(10)`를 통해 데이터의 앞부분만 출력한 예시입니다.</sub>

###### (1) 불필요한 컬럼 삭제

-   성별(Sex) 컬럼이 이미 존재하므로 Sex ID 컬럼을 삭제하였습니다.
-   연령대(Age Range) 컬럼이 존재하므로 Age Range ID 컬럼을 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   Percentage 컬럼의 데이터 총합이 100(%)과 일치함을 확인하였으며, 다른 이상치 또한 존재하지 않았습니다.

###### (4) 새로운 컬럼 추가

-   Age Range 컬럼의 정보를 바탕으로 '00s'~'80s' 까지 연령대를 나타내는 컬럼을 추가하였습니다.

<img src="../images/bike_pyramid_preprocessed_data.png">

##### 4) 할리스코 (Jalisco) 주의 성별에 따른 노동 인구 데이터

###### 📚Columns

-   State ID: 주(할리스코 주) 고유 id
-   State: 주 이름(할리스코)
-   Quarter ID: 분기 고유 id
-   Quarter: 연도별 분기
-   Sex ID: 성별의 고유 id
-   Sex: 성별
-   Workforce: 노동 인구
-   Workforce Total: 전체 노동 가능 인구
-   percentage: 전체 노동 가능 인구 대비 노동 인구의 비율

<img src="../images/bike_workforce_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   모든 데이터가 할리스코 (Jalisco) 주의 데이터에 해당하므로, State ID 및 State 컬럼을 삭제하였습니다.
-   분기(Quarter) 컬럼이 존재하므로 Quarter ID 컬럼을 삭제하였습니다.
-   마찬가지로 Sex ID 컬럼을 삭제하였으며, Time 컬럼은 분석에 불필요한 컬럼으로 판단하고 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   노동 인구 수가 전체 노동 가능 인구 수보다 많은 행이 있는지 검사하였으나 존재하지 않았습니다.
-   또한, 성별 컬럼에 남성/여성이 아닌 다른 값이 있거나 남성/여성 중 특정 분기에 누락된 성별이 있는지 검사하였고 역시 존재하지 않았습니다.

###### (4) 새로운 컬럼 추가

-   분기(Quarter)를 바탕으로 연도별로 구분한 Year 컬럼을 추가하였습니다.

<img src="../images/bike_workforce_preprocessed_data.png">

##### 5) 할리스코 (Jalisco) 주의 성별에 따른 통근 시간 데이터

###### 📚Columns

-   Sex ID: 성별 고유 id
-   Sex: 성별
-   Time to Work ID: 통근 시간 고유 id
-   Time to Work: 통근 시간
-   State ID: 주(할리스코 주) 고유 id
-   State: 주 이름(할리스코)
-   Population: 해당 조건에 맞는 인구 수
-   type: 성별
-   type ID: 성별 id
-   Share: 전체 대비 해당 조건에 맞는 인구의 비율

<img src="../images/bike_timetowork_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   성별(Sex) 컬럼이 존재하므로 Sex ID 컬럼을 삭제하였습니다.
-   통근 시간(Time to Work) 컬럼이 존재하므로 Time to Work ID 컬럼을 삭제하였습니다.
-   할리스코 (Jalisco) 주에 해당하는 데이터만 존재하므로 State 및 State ID 컬럼을 삭제하였습니다.
-   성별(Sex) 컬럼와 값이 중복되므로 type 및 type ID 컬럼을 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   범주형 데이터인 Time to Work 컬럼의 값이 남성/여성으로 각각 2개의 데이터가 문제 없이 존재함을 확인하였습니다.

<img src="../images/bike_timetowork_value_counts.png" width="200px">

-   또한 Share 컬럼의 값들은 남성/여성 데이터 모두 각각 총합이 1로 분포에 문제가 없었습니다.

<img src="../images/bike_timetowork_preprocessed_data.png" width="500px">

##### 6) 할리스코 (Jalisco) 주의 연령대 및 성별에 따른 세대주(Head of Household) 데이터

###### 📚Columns

-   Age Range ID: 연령대 고유 id
-   Age Range: 연령대
-   Sex ID: 성별 고유 id
-   Sex: 성별
-   Households: 해당 조건에 맞는 인구가 세대주인 가구 수
-   Households\_: Households에 음수를 취한 값

<img src="../images/bike_households_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   연령대(Age Range) 컬럼이 존재하므로 Age Range ID 컬럼을 삭제하였습니다.
-   성별(Sex) 컬럼이 존재하므로 Sex ID 컬럼을 삭제하였습니다.
-   Households\_ 컬럼은 불필요한 컬럼으로 판단하여 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   Age Range 컬럼에서 'No especificado'는 '특정되지 않음'을 나타내므로, 이상치로 간주하고 해당 데이터를 삭제하였습니다.

###### (4) 새로운 컬럼 추가

-   3번째 데이터셋과 마찬가지 방식으로, Age Range 컬럼의 정보를 바탕으로 '00s'~'80s' 까지 연령대를 나타내는 컬럼을 추가하였습니다.

<img src="../images/bike_households_preprocessed_data.png" width="400px">

##### 7) 할리스코 (Jalisco) 주의 통근 시간에 따른 통근 수단 이용 데이터

###### 📚Columns

-   Time to Work ID: 통근 시간 고유 id
-   Time to Work: 통근 시간
-   Work Mean ID: 통근 수단 고유 id
-   Work Mean: 통근 수단
-   Population: 해당 통근 시간 및 수단에 맞는 인구 수
-   Share: 해당 조건에 맞는 인구의 비율

<img src="../images/bike_workmeans_original_data.png">

###### (1) 불필요한 컬럼 삭제

-   통근 시간(Time to Work) 컬럼이 존재하므로 Time to Work ID 컬럼을 삭제하였습니다.
-   통근 수단(Work Mean) 컬럼이 존재하므로 Work Mean ID 컬럼을 삭제하였습니다.

###### (2) 중복행 및 결측치 검사

-   중복된 행이나 결측치가 존재하지 않았습니다.

###### (3) 이상치 검사

-   Time to Work 컬럼에 대해 `value_counts()`를 확인한 결과 분포 비중이 모두 동일하였습니다.
-   각 통근 수단 별 Share의 합이 모두 1로, 이상치가 존재하지 않았습니다.

<img src="../images/bike_workmeans_preprocessed_data.png" width="500px">

---

#### 🖥️ 2. 데이터 시각화 및 분석

-   막대그래프로 시각화한 결과 모든 연령대에서 남성 이용자의 비율이 여성 이용자의 비율보다 높았습니다.

<img src="../images/rental_counts_by_age_group.png">

-   연령대별 성별에 따른 이용자 비율을 보다 자세히 살펴보기 위해 성별에 따른 20~60대 이용자들의 비율을 파이 차트로 시각화하였습니다.

<img src="../images/rental_ratio_by_gender_of_age_groups.png">

-   남성 이용자의 비율이 더 높은 현상의 원인을 분석하기 위해 첫 번째 영가설을 수립하였습니다.

##### ⏬영가설 1: 남성과 여성의 이용자 비율 간 차이는 높은 성비 때문일 것이다.

-   연령대에 따른 남녀 성비를 조사하기 위해 해당 데이터셋이 다루는 ZMG 구역을 감싸는 멕시코의 할리스코(Jalisco) 주 인구 피라미드 데이터를 분석하였습니다.

<img src="../images/population_pyramid.png">

-   그러나 전 연령대에서 남성의 자전거 대여 서비스 이용 비율이 여성의 비율보다 높았던 것과 반대로,  
    실제 할리스코(Jalisco)의 인구 통계자료에 따르면 오히려 주요 연령대인 20~60대에서 여성의 비율이 더 높았습니다.

<img src="../images/gender_ratio_by_age_group.png">

##### ❌영가설 1은 참이 아니었음을 확인할 수 있습니다.

-   전 연령대에서 남성의 자전거 대여 서비스 이용 비율이 여성의 비율보다 높았던 것과 반대로, 실제 할리스코(Jalisco)의 인구 통계자료에 따르면 오히려 주요 연령대인 20~60대에서 여성의 비율이 더 높았습니다.

##### ➡️남성 자전거 이용 비율이 높은 현상은 단순히 성비 외에 복합적인 요인으로부터 기인했을 것이다.

-   새로운 영가설을 수립하기 위해 할리스코의 연령대 및 성별에 따른 세대주(Head of Household) 데이터를 시각화하였습니다.

<img src="../images/households_by_age_group_and_gender.png">

-   시각화 결과, 연령대별 성별에 따른 세대주 비율과 자전거 대여 서비스 이용객 비율은 대체로 유사한 분포를 보였습니다.
-   두 요인만으로는 상관관계를 정확히 파악하기 어려우므로, 할리스코 주의 성별에 따른 노동 인구 비율과 통근 시간 및 통근 수단 데이터를 추가로 분석하였습니다.

<img src="../images/workforce_total_by_year_and_sex.png">

-   성별에 따른 노동 인구 비율을 시각화한 결과, 2010년부터 2023년까지 전 연도에서 노동 가능 총 인구수는 여성이 남성보다 많았음에도 불구하고,  
    실제 노동자 수는 남성의 비율이 더 높게 나타났습니다.
-   현재까지의 분석 결과를 종합해보면, 할리스코의 남녀 성비는 여성이 남성보다 높은 비율을 차지하지만,  
    전 연령대에서 세대주 및 노동 인구 비율은 남성이 여성보다 높게 나타났습니다.
-   따라서 아래와 같은 영가설을 세워볼 수 있습니다.

##### ⏬영가설 2: 남성 자전거 이용 비율이 높은 현상은 할리스코 내 남성의 노동 비율이 더 높은 사실로부터 기인한다.

-   해당 가설을 검증하기 위해 할리스코 내의 통근 수단 데이터와 성별에 따른 통근 시간 데이터를 분석하였습니다.

<img src="../images/work_time_population_by_time_to_work.png">

-   할리스코 주의 통근 시간에 따라 이용하는 통근 수단 데이터를 시각화한 결과, 특히 15분 미만 그룹 및 15분 이상 30분 이하 그룹에서  
    자전거 이용 비율이 높게 나타났습니다.
-   또한 할리스코 주의 통근 시간 별 노동 인구 수 데이터를 시각화한 결과, 위 두 그룹에 해당하는 노동 인구 수가 가장 많았습니다.
-   따라서 가장 많은 노동 인구가 속하는 30분 이하 통근 시간 그룹에서, 특히 남성의 노동 비율이 여성보다 높은 해당 그룹에서 자전거 이용 비율이 높게  
    나타났다는 사실은 영가설 2를 뒷받침하는 근거가 될 수 있습니다.
-   추가적인 분석을 위해 자전거 대여 서비스 이용 데이터를 바탕으로 성별에 따른 총 대여 시간을 시각화하였습니다.

<img src="../images/ratio_by_rental_time_gender.png">

-   이때, 여성과 남성 모두 30분보다 오래 대여한 데이터의 비율이 2% 미만으로 낮았으므로, 4분위 분포도를 활용하여 중앙값으로 해당 데이터를 대체하였습니다.
-   앞선 분석에서 30분 이하 통근 시간을 가진 그룹, 그 중에서도 특히 노동 인구 비율이 높은 남성 그룹에서  
    자전거 이용 비율이 높게 나타났던 것과 더불어, 자전거 대여 데이터를 시각화한 결과로부터 남성과 여성 모두  
    약 98% 이상의 대여가 30분 이하로 이루어졌다는 점에서 영가설 2에 대한 근거를 찾을 수 있습니다.

-   출근 시간대와 퇴근 시간대에 대여 및 반납이 이루어진 데이터를 대상으로, 대여소의 위치를 바탕으로 지도에 시각화하였습니다.
-   출근 시간대는 07\~10시를 기준으로, 퇴근 시간대는 16\~19시를 기준으로 하였습니다.

<img src="../images/maps.png">

-   출퇴근 시간대 모두 대여 및 반납이 활발히 이루어진 과달라하라 시의 혁명 공원은 대로변에 위치하고 있어 버스 등 교통이 편리한 위치이며,  
    따라서 출퇴근 시 타 교통수단 이용 후 가까운 거리 이동을 위해 자전거 대여가 자주 이용되고 있는 것으로 예상됩니다.
-   출근 시간대에 반납이 주로 이루어진 할리스코 주의회 또한 대로변에 위치하고 있으며 중앙 광장이나 다양한 상점, 카페, 레스토랑 등이  
    주변에 자리하고 있어 근처로 출근하는 직장인들이 주로 이용하는 것으로 예상할 수 있습니다.

---

#### 📌 3. 결론

-   ZMG(과달라하라 대도시 구역)의 자전거 대여 데이터에서 **남성의 이용 비율이 여성보다 높게** 나타난 것은,
    **여성에 비해 노동 비율이 높은 남성이 주로 가까운 거리의 출퇴근 시 자전거 대여 서비스를 이용**한다는 사실 등 복합적인 요인이 기여한 결과로 예상됩니다.
