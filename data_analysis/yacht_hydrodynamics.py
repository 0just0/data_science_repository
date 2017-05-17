
# coding: utf-8

# # Лабораторная работа #1: Использование регрессии в scikit-learn
# 
# Используется набор данных "Yacht Hydrodynamics":
# https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics#

# ## Описание задачи
# Prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the ships' performance and for estimating the required propulsive power. Essential inputs include the basic hull dimensions and the boat velocity. 
# 
# 308 элементов выборки
# 
# Данные о геометрии яхт: 
# 
# 1. Longitudinal position of the center of buoyancy, adimensional.(Продольное положение центра плавучести)
# 2. Prismatic coefficient, adimensional.(Призматический коэффициент) 
# 3. Length-displacement ratio, adimensional.(Отношение длины перемещения)
# 4. Beam-draught ratio, adimensional. 
# 5. Length-beam ratio, adimensional. 
# 6. Froude number, adimensional. 
# 
# Измеряемая величина: 
# 
# 7. Residuary resistance per unit weight of displacement, adimensional.(Остаточное сопротивление)
# 

# 
# ## Первичная обработка данных и использование линейной регрессии
# 
# **Задача 1. Задача состоит в том, чтобы загрузить данные из csv файла, подготовить их для использования в scikit-learn и применить линейную регрессию (LinearRegression).**
# 
# 1.1. Загрузить данные из csv файла;
# 
# 1.2. Выполнить их предобработку (удалить отсутствующие значения, преобразовать категориальные переменные и т.д.);
# 
# 1.3. Преобразовать в матричную и векторную формы;
# 
# 1.4. Применить линейную регрессию, посчитать метрики (MAE, MSE, RMSE, R^2);
# 
# 1.5. Применить метод перекрестного тестирования и получить метрики при кросс-валидации.
# 
# 

# ### 1.1. Загрузка данных

# In[178]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# В качестве параметра передаем имя файла и разделитель - символ, которым проводится разбиение данных в файле
inputData = pd.read_csv('log.csv', sep=" ", header = None)
# Добавляем имена для колонок
inputData.columns = ["Longitudinal position of the center of buoyancy", "Prismatic coefficient", "Length-displacement ratio", "Beam-draught ratio", "Length-beam ratio", "Froude number", "Residuary number"]


# In[179]:

# Взглянем на таблицу
inputData


# Мы хотим предсказать на основании признаков остаточное сопротивление(Признак - Residuary number).

# In[180]:

# Имя столбца с целевой переменной
targetColumn = 'Residuary number'

# Получим имена всех столбцов и удалим оттуда целевой столбец
FeatureColumns = inputData.columns.tolist()
FeatureColumns.remove(targetColumn)


# ### 1.2. Предобработка данных

# В таблице могут содержаться пропущенные значения. Проверим, есть ли они в наших данных и если есть, то сколько их:

# In[181]:

print("Null values: {0}".format(inputData.isnull().values.any()))
print("Count of NaN values: {0}".format(np.sum(inputData.isnull().values)))



# В данном датасете пропущенные значения отсутствуют. 

# Посмотрим на типы значений, которые содержатся в наших колонках

# In[182]:

inputData.dtypes


# Итак, видно, что все данные в этом датасете числовые. Продолжаем подготовку данных.

# ### 1.3. Преобразование в матричную и векторную формы;

# In[183]:

from sklearn import preprocessing


# In[184]:

# Имя столбца с целевой переменной
targetColumn = 'Residuary number'

# Стандартизация данных
data_scaled = pd.DataFrame(preprocessing.scale(inputData))
data_scaled.columns = ["Longitudinal position of the center of buoyancy", "Prismatic coefficient", "Length-displacement ratio", "Beam-draught ratio", "Length-beam ratio", "Froude number", "Residuary number"]

# Получим имена всех столбцов и удалим оттуда целевой столбец
FeatureColumns = data_scaled.columns.tolist()
FeatureColumns.remove(targetColumn)
X = data_scaled[FeatureColumns].values
y = data_scaled[targetColumn].values


# ## Применение линейной регрессии

# In[185]:

from sklearn.linear_model import LinearRegression


# In[186]:

# После чего создадим объект класса и выполним подгон данных по всей выборке
lr = LinearRegression()

lr.fit(X,y) # Подгон данных


# После получения линейной регрессии, мы можем посмотреть коэффициенты и смещение:

# In[187]:

print("Intercept: ", lr.intercept_)
print("Coefficients: ", lr.coef_)


# Также, возпользовавшись стандартными функциями из пакета scikit-learn, посчитаем метрики:
# 

# In[188]:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Получим вектор "предсказаний"
y_predict = lr.predict(X)
# В функцию передается истинное значение вектора ответов и предсказанное нашей регрессионной функцией:
test_mae_error = mean_absolute_error(y, y_predict)
test_mse_error = mean_squared_error(y, y_predict)
test_rmse_error = mean_squared_error(y, y_predict)**0.5
test_r2_error = r2_score(y, y_predict)
print("MAE : {0}".format(test_mae_error))
print("MSE : {0}".format(test_mse_error))
print("RMSE: {0}".format(test_rmse_error))
print("R^2 coefficient : {0}".format(test_r2_error))


# In[189]:

plt.plot(y_predict, y, 'ro')
plt.title("Lin. Reg.  Corr=%f Rsq=%f" % (r2_score(y, y_predict), mean_squared_error(y, y_predict)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.plot([-3,4], [-3,4], 'b-')

plt.show()


# Из полученных параметров, можно сказать что 65% вариации объясняются изменением наших данных, судя по коэффициентам - больший вклад имеет Froude number(с коэффициентом 0.81009222).
# 
# Т.к. по данным параметрам сложно судить о качестве регрессии, применим метод перекрестной проверки, используя стандартные функции scikit-learn.

# In[ ]:

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)


# Теперь можем использовать перекрестную проверку:

# In[144]:

MAE_list_scores = []
MSE_list_scores = []
RMSE_list_scores = []
R2_list_scores = []

iteration_index = 0

# Разделение на тестовую и тренировочную выборки
for train_indexes, test_indexes in kf.split(X,y):
    iteration_index+=1
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    lr.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = lr.predict(X_test)
    x = np.array(X_test[:, -1]) 
    
    plt.plot(y_predict, y_test, 'ro')
    plt.title("Lin. Reg.  Corr=%f Rsq=%f" % (r2_score(y_test, y_predict), mean_squared_error(y_test, y_predict)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.plot([-3,4], [-3,4], 'g-')

    plt.show()
    
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = mean_squared_error(y_test, y_predict)**0.5
    current_r2 = r2_score(y_test, y_predict)
    print("Iteration #{0}: MAE : {1}, MSE : {2}, RMSE: {3} R2 : {4}".format(iteration_index, current_mae, current_mse, current_rmse, current_r2))
    MAE_list_scores.append(current_mae)
    MSE_list_scores.append(current_mse)
    RMSE_list_scores.append(current_rmse)
    R2_list_scores.append(current_r2)

# Выведем средние значения:
print("\nOverall: ")
print("\tMAE : {0}".format(np.mean(MAE_list_scores)))
print("\tMSE : {0}".format(np.mean(MSE_list_scores)))
print("\tRMSE : {0}".format(np.mean(RMSE_list_scores)))
print("\tR^2 coefficient : {0}".format(np.mean(R2_list_scores)))


# Посмотрим ошибку MSE регрессии, обученной на тестовой выборке и в результате перекрестной проверки:

# In[145]:

test_mse_error < np.mean(MSE_list_scores)


# Ожидаемо, MSE ошибка при обучении на всех данных оказалась меньше, чем на данных, при разбиении на тестовую и тренировочные данные.
# Но теперь, у нас есть некоторая степень уверенность в том, что алгоритм обладает какой-то обобщающей способностью.

# ** Задача 2. Полиномиальная регрессия **
# 
# 2.1. Применить полиномиальную регрессию и определить границы переобучения, т.е. при какой степени полиномиальной регрессии мы начинаем переобучаться?
# Привести значения степени, значения метрик для рассмотренных степеней и весовые коэффициенты регрессии (оптимальной степени).
# 

# In[159]:

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degreeList = []
maeList = []
mseList = []
rmseList = []

kf = KFold(n_splits=3, shuffle=True)

for count, degree in enumerate(range(0,9)):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    MAE_for_current_degree = []
    MSE_for_current_degree = []
    RMSE_for_current_degree = []
    
    for train_indexes, test_indexes in kf.split(X,y):
        # X_train, y_train - данные, соответствующие обучающей выборке
        X_train = X[train_indexes]
        y_train = y[train_indexes]
    
        # X_test, y_test - данные, соответствующие тренировочной выборке
        X_test = X[test_indexes]
        y_test = y[test_indexes]
    
        model.fit(X_train, y_train) # Обучение на тестовых данных
        y_predict = model.predict(X_test)
        
        current_mae = mean_absolute_error(y_test, y_predict)
        current_mse = mean_squared_error(y_test, y_predict)
        
        MAE_for_current_degree.append(current_mae)
        MSE_for_current_degree.append(current_mse)
        RMSE_for_current_degree.append(current_mse**0.5)
        
        plt.plot(y_predict, y_test, 'ro')
        plt.title("Lin. Reg.  Corr=%f Rsq=%f" % (r2_score(y_test, y_predict), mean_squared_error(y_test, y_predict)))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.plot([-4, 4],[-4, 4], 'g-')

        plt.show()
        
    
    print("Degree: {0}".format(degree))
    print("\tMAE : {0}".format(np.mean(MAE_for_current_degree)))
    print("\tMSE : {0}".format(np.mean(MSE_for_current_degree)))
    print("\tRMSE : {0}".format(np.mean(RMSE_for_current_degree)))
    degreeList.append(degree)
    maeList.append(np.mean(MAE_for_current_degree))
    mseList.append(np.mean(MSE_for_current_degree))
    rmseList.append(np.mean(RMSE_for_current_degree))


# In[160]:

plt.plot(degreeList, mseList, 'b-', label='MSE')
plt.title('MSE Error')
plt.ylabel('MSE Error')
plt.xlabel('Degree of polynomial')
plt.show()

plt.plot(degreeList, maeList, 'r', label='MAE')
plt.title('MAE Error')
plt.ylabel('MAE Error')
plt.xlabel('Degree of polynomial')
plt.show()

plt.plot(degreeList, rmseList, 'r', label='RMSE')
plt.title('RMSE Error')
plt.ylabel('RMSE Error')
plt.xlabel('Degree of polynomial')
plt.show()


# ### Итог:
# Исходя из значений коэффициентов и приведенных выше графиков, на третьей степени уже наступает переобучение, а при первой ошибка достаточно высока, поэтому следует выбрать вторую степень(Смотрим коэффициенты MSE и RMSE).

# In[161]:

model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train, y_train) # Обучение на тестовых данных
y_predict = model.predict(X_test)


# Теперь получим значения коэффициентов:

# In[162]:

print("Coefficients: {0}".format((model.get_params()['linearregression']).coef_))
print("Intercept: {0}".format((model.get_params()['linearregression']).intercept_ ))


# **Контрольный вопрос: запишите уравнение линейной регрессии (возможно полиномиальной), на котором для Вашей задачи достигается минимум MSE ошибки (на перекрестной проверке):**

# Запишем уравнение для линейной регрессии в первой степени:

# In[163]:

model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(X_train, y_train) # Обучение на тестовых данных
y_predict = model.predict(X_test)
print("Coefficients: {0}".format((model.get_params()['linearregression']).coef_))
print("Intercept: {0}".format((model.get_params()['linearregression']).intercept_ ))


# Y = 0 + 0.02256657*$x_1$ + -0.00171752*$x_2$ + 0.08475665*$x_3$ + -0.08716871*$x_4$ + -0.09047215*$x_5$ + 0.8446997*$x_6$

# ## $L_1$ и $L_2$ регрессия
# 
# **Задача 3. Применить и сравнить $L_1$ и $L_2$ регрессии с оптимальным значением степени, полученной в результате выполнения задачи 2. Получить метрики: MAE, MSE, RMSE и значения весовых коэффициентов.** 
# 
# Теперь, когда нам удалось выяснить границы переобучаемости для полиномиальной регрессии, попробуем применить различные типы регрессии к нашим данным.
# 
# Начнем рассмотрение $L_2$ регрессии (ридж-регрессии) с параметрами по умолчанию.
# 
# Регрессия в scikit-learn реализуется в сл. модулях:
#     sklearn.linear_model.Ridge ( http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn-linear-model-ridge )
#     sklearn.linear_model.Lasso ( http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso )

# In[164]:

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Мы выяснили, что оптимальным значением степени для полиномиальной регрессии является 2, поэтому укажем её.
optimalDegree = 2

# Будем использовать параметры по умолчанию, т.е. на данный момент не укажем значение alpha
model = make_pipeline(PolynomialFeatures(2), Ridge())
kf = KFold(n_splits=5, shuffle=True)

MAE_list = []
MSE_list = []
RMSE_list = []

for train_indexes, test_indexes in kf.split(X,y):
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    model.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = model.predict(X_test)
        
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = current_mse**0.5
        
    MAE_list.append(current_mae)
    MSE_list.append(current_mse)
    RMSE_list.append(current_rmse)
print("\tMAE : {0}".format(np.mean(MAE_list)))
print("\tMSE : {0}".format(np.mean(MSE_list)))
print("\tRMSE : {0}".format(np.mean(RMSE_list)))

print("Coefficients: {0}".format((model.get_params()['ridge']).coef_))
print("Intercept: {0}".format((model.get_params()['ridge']).intercept_ ))


# ** Сравните и объясните полученные значения MSE и весовых коэффициентов $L_2$-регрессии со значениями линейной регрессии**
# 
# С помощью $L_2$-регрессии мы смогли получить меньшее значение среднего квадрата ошибки по сравнению с полиномиальной.
# Так же понизились значения части коэффициентов, что и следовалао ожидать, т.к. часть наших данных зависят друг от друга.(Это следует и зопределения Ридж-регрессии)

# Теперь рассмотрим $L_1$ регрессию (с параметрами по умолчанию).

# In[165]:

model = make_pipeline(PolynomialFeatures(2), Lasso(max_iter=1e5))
kf = KFold(n_splits=5, shuffle=True)

MAE_list = []
MSE_list = []
RMSE_list = []

for train_indexes, test_indexes in kf.split(X,y):
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    model.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = model.predict(X_test)
        
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = current_mse**0.5
        
    MAE_list.append(current_mae)
    MSE_list.append(current_mse)
    RMSE_list.append(current_rmse)
print("\tMAE : {0}".format(np.mean(MAE_list)))
print("\tMSE : {0}".format(np.mean(MSE_list)))
print("\tRMSE : {0}".format(np.mean(RMSE_list)))

print("Coefficients: {0}".format((model.get_params()['lasso']).coef_))
print("Intercept: {0}".format((model.get_params()['lasso']).intercept_ ))


# ** Сравните и объясните полученные значения MSE и весовых коэффициентов $L_1$-регрессии со значениями линейной регрессии**
# 
#    В нашем случае, Лассо-регрессия загуляет все коэффициенты, т.к. дефолтная alpha = 1 слишком велика, при уменьшении данного параметра мы получаем наиболее значимые параметры и снижаем ошибку.
# 
# ** Какие значения были отброшены моделью, в процессе $L_1$-регуляризации? **
#    Все(см. выше)
# 
# ** Сравните и объясните полученные значения MSE и весовых коэффициентов $L_1$-регрессии со значениями $L_2$-регрессии**
# 
# ** Что лучше использовать для данной задачи: $L_1$ или $L_2$ регрессию?**
#     Используя функции без указания гипермараметра, очевидно, что лучше использовать Ridge-регрессию, т.к мы получаем определенные значимые результаты.

# ### Подбор гиперпараметра в задаче регрессии
# 
# ** Задача 4. Подобрать гиперпараметры для $L_2$ и $L_1$ регрессии. Получить значения метрик MAE, MSE, RMSE и весовых коэффициентов для оптимального параметра. Объяснить полученные результаты.**

# In[166]:

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

MAE_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
MSE_scorer = make_scorer(mean_squared_error, greater_is_better=False)


# In[167]:

n_alphas = 200 # 200 значений
alphas = np.logspace(-10, 2, n_alphas)
alphas


# In[168]:

#parameters = {'alpha':alphas}
parameters = {'alpha': alphas, 'fit_intercept' : [True, False]}


# In[169]:

clf_ridge = GridSearchCV(Ridge(), parameters, scoring  = MSE_scorer)
clf_ridge.fit(X, y)


# In[170]:

clf_lasso = GridSearchCV(Lasso(max_iter=1e5), parameters, scoring  = MSE_scorer)
clf_lasso.fit(X, y)


# In[171]:

clf_ridge.best_params_


# In[172]:

clf_lasso.best_params_


# In[173]:

clf_ridge.best_score_


# In[174]:

clf_lasso.best_score_


# In[175]:

clf_ridge.best_estimator_


# In[176]:

clf_lasso.best_estimator_


# #### Итог:
# 
# Мы смогли получить оптимальные параметры alpha, при которых наша модель ведет себя наилучшим образом. Теперь используем их для оценки подходящих коэффициентов.

# #### Ridge-regression

# In[ ]:

# Мы выяснили, что оптимальным значением степени для полиномиальной регрессии является 2, поэтому укажем её.
optimalDegree = 2

# передаем лучший estimator, подобранный с помощью GridSearchCV
model_ridge = make_pipeline(PolynomialFeatures(2), clf_ridge.best_estimator_)
kf = KFold(n_splits=5, shuffle=True)

MAE_list = []
MSE_list = []
RMSE_list = []

for train_indexes, test_indexes in kf.split(X,y):
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    model_ridge.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = model_ridge.predict(X_test)
        
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = current_mse**0.5
        
    MAE_list.append(current_mae)
    MSE_list.append(current_mse)
    RMSE_list.append(current_rmse)
print("\tMAE : {0}".format(np.mean(MAE_list)))
print("\tMSE : {0}".format(np.mean(MSE_list))) 
print("\tRMSE : {0}".format(np.mean(RMSE_list))) 

print("Coefficients: {0}".format((model_ridge.get_params()['ridge']).coef_))
print("Intercept: {0}".format((model_ridge.get_params()['ridge']).intercept_ ))


# #### Lasso-regression

# In[ ]:

# Мы выяснили, что оптимальным значением степени для полиномиальной регрессии является 2, поэтому укажем её.
optimalDegree = 2

# передаем лучший estimator, подобранный с помощью GridSearchCV
model_lasso = make_pipeline(PolynomialFeatures(2), clf_lasso.best_estimator_)
kf = KFold(n_splits=5, shuffle=True)

MAE_list = []
MSE_list = []
RMSE_list = []

for train_indexes, test_indexes in kf.split(X,y):
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    model_lasso.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = model_lasso.predict(X_test)
        
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = current_mse**0.5
        
    MAE_list.append(current_mae)
    MSE_list.append(current_mse)
    RMSE_list.append(current_rmse)
print("\tMAE : {0}".format(np.mean(MAE_list)))
print("\tMSE : {0}".format(np.mean(MSE_list)))
print("\tRMSE : {0}".format(np.mean(RMSE_list)))

print("Coefficients: {0}".format((model_lasso.get_params()['lasso']).coef_))
print("Intercept: {0}".format((model_lasso.get_params()['lasso']).intercept_ ))


# **Задача 4.2 Подбор гиперпараметров для L1 или L2 полиномиальной регрессии (со значением степени полученной из задачи 2). Получить значения метрик MAE, MSE, RMSE и весовых коэффициентов для оптимального параметра. Объяснить полученные результаты.**
# 
# Взглянем на модель конвейера:

# In[ ]:

model_ridge


# In[ ]:

parameters = {'ridge__alpha': alphas, 'ridge__fit_intercept' : [True, False]}


# In[ ]:

clf_ridge = GridSearchCV(model_ridge, parameters, scoring  = MSE_scorer)
clf_ridge.fit(X, y)


# In[ ]:

clf_ridge.best_params_


# In[ ]:

clf_ridge.best_score_


# In[ ]:

# передаем лучший estimator, подобранный с помощью GridSearchCV
kf = KFold(n_splits=5, shuffle=True)

MAE_list = []
MSE_list = []
RMSE_list = []
new_model = clf_ridge.best_estimator_

for train_indexes, test_indexes in kf.split(X,y):
    # X_train, y_train - данные, соответствующие обучающей выборке
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    # X_test, y_test - данные, соответствующие тренировочной выборке
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    new_model.fit(X_train, y_train) # Обучение на тестовых данных
    y_predict = new_model.predict(X_test)
        
    current_mae = mean_absolute_error(y_test, y_predict)
    current_mse = mean_squared_error(y_test, y_predict)
    current_rmse = current_mse**0.5
        
    MAE_list.append(current_mae)
    MSE_list.append(current_mse)
    RMSE_list.append(current_rmse)
    
print("\tMAE : {0}".format(np.mean(MAE_list)))
print("\tMSE : {0}".format(np.mean(MSE_list)))
print("\tRMSE : {0}".format(np.mean(RMSE_list)))

print("Coefficients: {0}".format((new_model.get_params()['ridge']).coef_))
print("Intercept: {0}".format((new_model.get_params()['ridge']).intercept_ ))

