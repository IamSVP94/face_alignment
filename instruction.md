# Работа с репозиторием

1. Разработка велась на `python3.10`. Нужно установить зависимости из `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

2. Нужно создать выборку для обучения и валидации. Нужно запустить след. скрипт:

```bash
python3 make_dataset_crops_1.py --src_dir=src_dir
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
tensorboard --logdir=logs
```

4. Далее требуется сформировать датасет из предсказаний на обученной сетке и с помощью dlib.

```bash
python3 pred_pl_3.py  --src_dir=Menpo/test/ --checkpoint_pl=checkpoint.ckpt
```
, где `Menpo/test/` - папка с test изображениями, `checkpoint.ckpt` - чекпоинт обученной модели в `pytorch_lightning`

```bash
python3 dlib_pred_3.5.py -s Menpo/test/ -m shape_predictor_68_face_landmarks.dat
```
, где `Menpo/test/` - папка с test изображениями, `shape_predictor_68_face_landmarks.dat` - веса dlib.

5. Далее следует запустить скрипт расчета и отображения метрики CED

```bash
python3 count_ced_for_points_4.py --gt_path=Menpo/test/ --predictions_path=Menpo/test_preds/ --dlib_path=Menpo/test_dlib/ --output_path=Menpo/Menpo.png
```