## AMO_1
# Содержание задания:
Необходимо из “подручных средств” создать простейший конвейер для автоматизацииработы с моделью машинного обучения на Linux. Отдельные этапы конвейера машинногообучения описываются в разных python–скриптах, которые потом соединяются(иногдаиспользуют термин “склеиваются”) с помощью bash-скрипта.
* Для запуска конвеера необходимо клонировать реппозиторий и запустить файл pipeline.sh
* Файл data_creation.py создает данные (насторен создавать кубическую зависимость х от у с шумами). При запуске создаются одновременно 4 файла тестовых и 4 тренировочных с рандомным уровнем шума.
* model_preprocessing - выполняет предобработкуданных,  с помощью sklearn.preprocessing.StandardScaler и создает в файлах тестовых и тренировочных дополнительную колонку со стандартизованными значениями X.
* model_preparation - этот файл складывает все тренировочные данные в один датасет, применяет полимониальную трансформацию и тренерует линейную регрессию, модель сохраняется в файл.
* model_testing - выполняет пресказания на основе полученной модели и записывает данные о среднеквадратичном отклонении в файл score.txt.
