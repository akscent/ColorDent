# Разработка ML алгоритма для приложения ColorDent 

[^1] подробнее о **ColorDent**

[^1]: **ColorDent** - мобильное приложение-помощник стоматолога для подбора коректной цветопробы пломбировочного материала на основе фотографии пациента.

В этом репозитории представлен код эмпирического сравнения различных алгоритмов обучения с учителем для автоматической детекции зубов человека на фотографии и мульти-классификации их оттенков. Выводится один общий подход, включающий: 

1) Чтение фотографий классом [ReadImages.py](Experiment/Source/ReadImages.py)

> Пример начального фото 
<details>
  <summary>Показать оригинальное изображение</summary>

![Оригинальное фото](img/orig/101_0001.jpg)
</details>

2) Препроцессинг фотографий через класс [PreProcessingData.py](Experiment/Source/PreProcessingData.py)

> Детекция нужного участка фото
<details>
  <summary>Показать обрезанное изображение</summary>

![Обрезанное фото](img/teeth/101_0001.JPG)
</details>

> Различные канальные преобразования, например:
<details>
  <summary>Показать HSV</summary>

![HSV канал](img/features/101_0001.JPG)
</details>

> Набор канальных изображений:
<details>
  <summary>Показать другие канальные преобразования</summary>

![barchar_stack](img/features/BAR_StackColor_101_0001.JPG)
</details>

> Гистограммы для различных каналов:

<details>
  <summary>Показать гистограммы</summary>

![barchar_](img/features/BAR_CHART_HUE_101_0001.JPG)
![barchar_](img/features/BAR_CHART_SATURATION_101_0001.JPG)
![barchar_](img/features/BAR_CHART_VALUE_101_0001.JPG)
</details>

3) Детекция зубов классом [TeethDetection.py](Experiment/Source/TeethDetection.py)

<details>
  <summary>Показать маску</summary>

![mask](img/features/MASKOVERLAY_101_0001.JPG)
</details>

4) После всех преобразований вычисляются фичи и записываются в таблицы напротив соответствующих лейблов с помощью класса [FeatureExtraction.py](Experiment/Source/FeatureExtraction.py)

5) Наконец в классе [Classification.py](Experiment/Source/Classification.py) используется ансамбль моделей для классификации изображений по лейблам, соответствующим цвету определенного производителя. `Accuracy` ~ 92,8 %

## TO DO 

Необходимо добавить: 
* Мультиклассовая классификация изображений в зависимости от цветовой схемы различных производителей пломб.