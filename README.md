### Задание

Реализация алгоритма обнаружения 68 особых точек на лице человека

Задание заключается в реализации алгоритма обнаружения 68 особых точек на лице человека (face alignment), тестировании
данного алгоритма на общедоступных датасетах и сравнении с аналогами.

Обучающие данные представляют собой разметку 68 точек лица, в формате IBUG (https://ibug.doc.ic.ac.uk/resources/300-W/).
Данные для обучения и тестирования можно найти по
ссылке https://drive.google.com/file/d/0B8okgV6zu3CCWlU3b3p4bmJSVUU/view?usp=sharing. Предоставленные данные состоят из
датасетов 300W и Menpo, на тестовой части которых (папки test) нужно замерить точность работы обученной модели.
Обучающие данные можно использовать любые, например можно взять train часть 300W и Menpo датасетов (из Menpo нужно
выкинуть профильные изображения, на которых размечено только 39 точек).

В качестве прямоугольника лица нужно использовать прямоугольник, выдаваемый детектором лиц из библиотеки
DLIB (https://github.com/davisking/dlib/blob/master/python_examples/face_detector.py). При необходимости ограничивающий
прямоугольник из dlib можно модифицировать.

За основу алгоритма можно (но не обязательно) взять архитектуру нейронной сети, описанной в статье "Joint Face Detection
and Alignment using Multi-task Cascaded Convolutional Networks" (https://arxiv.org/pdf/1604.02878.pdf, сеть ONet,
fig.2), оставив в функции потерь только член, отвечающий за особые точки лица.

#### В отчёте по результатам выполнения тестового задания должны быть:

- [x] код для обучения и тестирования модели

- [x] скрипт, позволяющий минимальными усилиями воспроизвести результат (запустить обучение и тестирование) и
  документация к нему

- [ ] отчёт в формате pdf с описанием проделанных экспериментов (какие подходы были проверены, какие результаты и выводы
  были сделаны и т.д.). В отчёте помимо описания проделанных экспериментов нужно вставить графики по метрике CED (
  cummulative error distribution), по оси X в которых идёт нормированная среднеквадратичная ошибка, а по оси Y - процент
  изображений в тестовой выборке, ошибка на которых меньше значения по оси X. Максимальное значение ошибки по оси X
  взять равным 0.08. Нормализацию ошибки нужно производить на величину sqrt(H * W), где H, W - высота и ширина
  прямоугольника лица, полученного с помощью DLIB. Ошибку нужно считать в координатах исходного изображения. Для каждого
  графика CED необходимо посчитать площадь под кривой - это будет целевая метрика для сравнения алгоритмов. Графики CED
  должны быть построены отдельно для датасетов 300W и Menpo. Помимо предложенного метода нужно отобразить на графике
  точность работы детектора точек DLIB для базы Menpo (детектор dlib обучался на базе 300W, поэтому тестировать dlib на
  ней не имеет смысла).

Пример использования :
http://dlib.net/face_landmark_detection.py.html

<p>Пример скрипта для подсчёта CED графиков можно найти тут:
https://drive.google.com/file/d/0B8okgV6zu3CCTk96SW9IWFJ6RE0/view?usp=sharing
Минимальный результат - предложенный алгоритм должен быть близок к DLIB по площади под CED графиком при пороге 0.08.</p>

---

### Работа с репозиторием

1. Разработка велась на `python3.10`. Нужно установить зависимости из `requirements.txt`:

```bash
python3 -m pip install - r requirements.txt
```

2. Нужно создать выборку для обучения и валидации. Нужно запустить след. скрипт:

```bash
python3 make_dataset_crops_1.py -src_dir=src_dir
```

, где `src_dir` - папка с исходными изобр. и разметкой в формате *.pts
Для лучшей навигации складываем обработанные папки */train_dataset и */test_dataset из 300W и Menpo вместе.

P.S. Вывести результат на экран можно с помощью команды

```bash
python3 show_drawed_points.py --src_dir=test/
```

, где `test` - папка с изображениями и разметкой

3. На данной выборке запустить обучение:

```bash
python3 train_pl_2.py --train_dir=train_dir/ --val_dir=val_dir/ --epochs=100 --batch_size=512 --device=cuda
```

, где - `train_dir/` - общая папка с `300W/train/` `Menpo/train/`, `val_dir/` - общая папка с `300W/test/` `Menpo/test/`
и
настройки гиперпараметров обучения.

За процессом обучения можно следить с помощью графиков в `tensorboard`:

```bash
tensorboard --logdir logs/
```

4. Далее требуется сформировать датасет из предсказаний на обученной сетке.

```bash
python3 pred_pl_3.py  --src_dir=300W/test/ --checkpoint_pl=checkpoint.ckpt
```

, где `300W/test/` - папка с test изображениями, `checkpoint.ckpt` - чекпоинт обученной модели в `pytorch_lightning`

5. Далее следует запустить скрипт расчета и отображения метрики CED

```bash
python3 count_ced_for_points_4.py --gt_path=300W/test/ --predictions_path=300W/test_preds/ --output_path=300W/test.png
```