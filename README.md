# ErrorCorrector
Python переделки [матлаб](https://github.com/Mirkes/Error-corrector) и [c++](https://github.com/olgashemagina/BAISim) версии.

### TODO:
- [x] Минимальное обучение нейронки (resnet18) + возможность дообучить
- [x] Формирование выборки 1 класс - ошибки (признаки), 2 класс - правильные результаты (признаки) + центрирование 
- [ ] применение метода главных компонент ([PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)) на обучающей выборке, включающей оба класса (PCA на все данные)
    - [ ] количество главных компонент
    - [ ] [whitening transformation](http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/)
    - [ ] построение проекции данных на главные компоненты
- [ ] Установка числа кластеров для разделения ошибок на группы с помощью алгоритма k- средних (k-means)
- [ ] Построение дискриминанта Фишера
- [ ] Строим графики