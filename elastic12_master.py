from elasticsearch import Elasticsearch, helpers
from faker import Faker
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
import csv
import codecs
import numpy as np


# Расширенная настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('elasticsearch_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Инициализация Elasticsearch
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    verify_certs=False,
    max_retries=5,
    retry_on_timeout=True,
    timeout=300
)

index_name = "baza10"
fake = Faker('ru_RU')

def check_elasticsearch_health():
    """Проверка здоровья кластера Elasticsearch"""
    """Проверка здоровья кластера Elasticsearch"""
    try:
        health = es.cluster.health()
        logging.info(f"Elasticsearch cluster health: {health['status']}")
        return health['status'] in ['green', 'yellow']
    except Exception as e:
        logging.error(f"Failed to check Elasticsearch health: {e}")
        return False

def create_index():
    """Создание индекса с динамическим маппингом"""
    try:
        if not check_elasticsearch_health():
            raise Exception("Elasticsearch cluster is not healthy")

        if es.indices.exists(index=index_name):
            logging.info(f"Индекс {index_name} существует, удаляем...")
            es.indices.delete(index=index_name)
            time.sleep(2)

        index_settings = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s"
                }
            },
            "mappings": {
                "dynamic": True  # Разрешаем динамическое создание полей
            }
        }

        es.indices.create(index=index_name, body=index_settings)
        logging.info(f"Индекс {index_name} создан успешно.")
        return True
    except Exception as e:
        logging.error(f"Ошибка при создании индекса: {e}")
        messagebox.showerror("Ошибка", f"Не удалось создать индекс: {str(e)}")
        return False

def preview_csv_file(file_path, encoding, delimiter):
    """Показать первые 5 строк CSV файла"""
    try:
        with codecs.open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            return [delimiter.join(next(reader)) for _ in range(5)]
    except Exception as e:
        return [f"Ошибка предпросмотра: {str(e)}"]


def preprocess_data(row_dict):
    """
    Заменяет NaN значения на пробел в словаре данных,
    гарантируя строковое представление для Elasticsearch
    """
    processed = {}
    for key, value in row_dict.items():
        # Проверяем различные варианты NaN значений
        if pd.isna(value) or str(value) == 'nan' or str(value) == 'NaN' or value == 'NaN':
            processed[key] = " "  # пробел как строка
        else:
            # Преобразуем все значения в строки, кроме чисел
            if isinstance(value, (int, float)):
                processed[key] = value
            else:
                processed[key] = str(value)
    return processed


def import_csv_in_batches(file_path, encoding, delimiter, skip_first=True, batch_size=100):
    """
    Импорт данных из CSV файла в Elasticsearch батчами по 100 записей.
    Все поля импортируются как текст, пустые значения заменяются на пробел.

    Args:
        file_path (str): Путь к CSV файлу
        encoding (str): Кодировка файла
        delimiter (str): Разделитель полей
        skip_first (bool): Пропустить первую строку (заголовки)
        batch_size (int): Размер батча для импорта
    """
    try:
        with codecs.open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Читаем первую строку для определения количества колонок
            first_row = next(reader)
            num_columns = len(first_row)

            # Генерируем заголовки, если первая строка не содержит их
            # или если они пустые
            headers = []
            using_generated_headers = False

            if all(not header.strip() for header in first_row):
                # Если все заголовки пустые, генерируем новые
                headers = [f'col{i + 1}' for i in range(num_columns)]
                using_generated_headers = True
            else:
                # Проверяем каждый заголовок и заменяем пустые на сгенерированные
                for i, header in enumerate(first_row):
                    if not header.strip():
                        headers.append(f'col{i + 1}')
                        using_generated_headers = True
                    else:
                        headers.append(header.strip())

            # Если skip_first=True и мы не сгенерировали заголовки,
            # пропускаем следующую строку
            if skip_first and not using_generated_headers:
                next(reader)

            batch = []
            total_processed = 0

            for row in reader:
                # Создаем словарь, заменяя пустые значения на пробел
                document = {}
                for i, value in enumerate(row):
                    if i < len(headers):
                        # Заменяем пустые значения и nan на пробел
                        cleaned_value = value.strip()
                        if not cleaned_value or cleaned_value.lower() == 'nan':
                            cleaned_value = ' '
                        document[headers[i]] = cleaned_value

                batch.append(document)

                # Если набрали батч - импортируем
                if len(batch) >= batch_size:
                    try:
                        actions = [
                            {
                                "_index": index_name,
                                "_source": doc
                            }
                            for doc in batch
                        ]
                        helpers.bulk(es, actions)
                        total_processed += len(batch)
                        logging.info(f"Импортировано {total_processed} записей")

                        # Очищаем батч
                        batch = []

                    except Exception as e:
                        logging.error(f"Ошибка при импорте батча: {e}")
                        continue

            # Импортируем оставшиеся записи
            if batch:
                try:
                    actions = [
                        {
                            "_index": index_name,
                            "_source": doc
                        }
                        for doc in batch
                    ]
                    helpers.bulk(es, actions)
                    total_processed += len(batch)
                    logging.info(f"Импортировано {total_processed} записей")

                except Exception as e:
                    logging.error(f"Ошибка при импорте последнего батча: {e}")

            return total_processed

    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        raise


def import_csv_dialog():
    """Диалог настроек импорта CSV"""
    dialog = tk.Toplevel()
    dialog.title("Импорт CSV")
    dialog.geometry("600x450")  # Увеличил высоту для нового поля
    dialog.grab_set()  # Делаем окно модальным

    # Настройки импорта
    settings_frame = ttk.LabelFrame(dialog, text="Настройки импорта")
    settings_frame.pack(fill=tk.X, padx=5, pady=5)

    # Frame для строки с названием индекса
    index_frame = ttk.Frame(settings_frame)
    index_frame.pack(fill=tk.X, padx=5, pady=2)

    # Поле для ввода названия индекса
    ttk.Label(index_frame, text="Название индекса:").pack(side=tk.LEFT, padx=5)
    index_name_var = tk.StringVar(value=index_name)
    index_entry = ttk.Entry(index_frame, textvariable=index_name_var, width=30)
    index_entry.pack(side=tk.LEFT, padx=5)

    # Frame для первой строки настроек
    settings_row1 = ttk.Frame(settings_frame)
    settings_row1.pack(fill=tk.X, padx=5, pady=2)

    # Кодировка
    encoding_var = tk.StringVar(value="utf-8")
    ttk.Label(settings_row1, text="Кодировка:").pack(side=tk.LEFT, padx=5)
    encoding_combo = ttk.Combobox(settings_row1, textvariable=encoding_var,
                                  values=["utf-8", "cp1251"], width=10)
    encoding_combo.pack(side=tk.LEFT, padx=5)

    # Разделитель
    ttk.Label(settings_row1, text="Разделитель:").pack(side=tk.LEFT, padx=5)
    delimiter_var = tk.StringVar(value=",")
    delimiter_choices = [
        (",", "Запятая (,)"),
        (";", "Точка с запятой (;)"),
        ("\t", "Табуляция (\\t)"),
        ("|", "Вертикальная черта (|)"),
        ("custom", "Другой...")
    ]

    def on_delimiter_change(*args):
        if delimiter_combo.get() == "Другой...":
            custom_delimiter_entry.pack(side=tk.LEFT, padx=5)
        else:
            custom_delimiter_entry.pack_forget()
            # Установить выбранный разделитель
            for delim, name in delimiter_choices:
                if name == delimiter_combo.get():
                    delimiter_var.set(delim)
                    break

    delimiter_combo = ttk.Combobox(settings_row1,
                                   values=[name for _, name in delimiter_choices],
                                   width=20)
    delimiter_combo.set("Запятая (,)")
    delimiter_combo.pack(side=tk.LEFT, padx=5)
    delimiter_combo.bind('<<ComboboxSelected>>', on_delimiter_change)

    # Поле для пользовательского разделителя
    custom_delimiter_entry = ttk.Entry(settings_row1, width=5)
    custom_delimiter_entry.bind('<KeyRelease>',
                                lambda e: delimiter_var.set(custom_delimiter_entry.get()))

    # Frame для второй строки настроек
    settings_row2 = ttk.Frame(settings_frame)
    settings_row2.pack(fill=tk.X, padx=5, pady=2)

    # Пропуск первой строки
    skip_first = tk.BooleanVar(value=True)
    ttk.Checkbutton(settings_row2, text="Пропустить первую строку",
                    variable=skip_first).pack(side=tk.LEFT, padx=5)

    # Использовать первую строку как заголовки
    use_headers = tk.BooleanVar(value=True)
    ttk.Checkbutton(settings_row2, text="Использовать первую строку как заголовки",
                    variable=use_headers).pack(side=tk.LEFT, padx=5)

    # Frame для третьей строки настроек
    settings_row3 = ttk.Frame(settings_frame)
    settings_row3.pack(fill=tk.X, padx=5, pady=2)

    # Количество строк
    ttk.Label(settings_row3, text="Количество строк (пусто = все):").pack(side=tk.LEFT, padx=5)
    rows_entry = ttk.Entry(settings_row3, width=10)
    rows_entry.pack(side=tk.LEFT, padx=5)

    # Предпросмотр
    preview_frame = ttk.LabelFrame(dialog, text="Предпросмотр")
    preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Добавляем скроллбары для предпросмотра
    preview_scroll_y = ttk.Scrollbar(preview_frame)
    preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    preview_scroll_x = ttk.Scrollbar(preview_frame, orient='horizontal')
    preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    preview_text = tk.Text(preview_frame, height=10, wrap=tk.NONE,
                           yscrollcommand=preview_scroll_y.set,
                           xscrollcommand=preview_scroll_x.set)
    preview_text.pack(fill=tk.BOTH, expand=True)

    preview_scroll_y.config(command=preview_text.yview)
    preview_scroll_x.config(command=preview_text.xview)

    filename = [None]  # Используем список для хранения имени файла

    def select_file():
        filename[0] = filedialog.askopenfilename(filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ])
        if filename[0]:
            current_delimiter = delimiter_var.get()
            preview_rows = preview_csv_file(filename[0], encoding_var.get(), current_delimiter)
            preview_text.delete(1.0, tk.END)
            for row in preview_rows:
                preview_text.insert(tk.END, f"{row}\n")

    def start_import():
        if not filename[0]:
            messagebox.showerror("Ошибка", "Выберите файл для импорта")
            return

        # Проверка названия индекса
        custom_index = index_name_var.get().strip()
        if not custom_index:
            messagebox.showerror("Ошибка", "Введите название индекса")
            return

        try:
            # Создаем индекс с указанным именем
            global index_name
            index_name = custom_index
            if not create_index():
                return

            # Создание прогресс-бара
            progress_window = tk.Toplevel()
            progress_window.title("Импорт CSV")
            progress_window.geometry("300x150")

            progress_label = ttk.Label(progress_window, text="Импорт данных...")
            progress_label.pack(pady=10)

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(pady=10, padx=20, fill=tk.X)

            status_label = ttk.Label(progress_window, text="0 записей импортировано")
            status_label.pack(pady=10)

            def update_progress(current_count):
                status_label.config(text=f"{current_count} записей импортировано")
                progress_window.update()

            # Импорт данных
            total_imported = import_csv_in_batches(
                filename[0],
                encoding_var.get(),
                delimiter_var.get(),
                skip_first.get()
            )

            progress_window.destroy()
            messagebox.showinfo("Успех", f"Импортировано {total_imported} записей в индекс {index_name}")
            dialog.destroy()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка импорта: {str(e)}")
            logging.error(f"Import error: {str(e)}")

    # Кнопки
    button_frame = ttk.Frame(dialog)
    button_frame.pack(fill=tk.X, pady=5)

    ttk.Button(button_frame, text="Выбрать файл", command=select_file).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Импортировать", command=start_import).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

def export_to_xlsx():
    """Экспорт результатов поиска в XLSX"""
    if not tree.get_children():
        messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
        return

    filename = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )
    if filename:
        try:
            data = []
            for item in tree.get_children():
                data.append([tree.item(item)['values'][i] for i in range(len(columns))])

            df = pd.DataFrame(data, columns=columns)
            df.to_excel(filename, index=False)
            messagebox.showinfo("Успех", "Данные экспортированы успешно")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте: {str(e)}")
            logging.error(f"Export error: {str(e)}")


def select_index():
    """Выбор индекса для поиска с возможностью удаления"""
    try:
        indices = list(es.indices.get_alias().keys())
        dialog = tk.Toplevel()
        dialog.title("Управление индексами")
        dialog.geometry("400x500")
        dialog.grab_set()  # Делаем окно модальным

        # Frame для списка и кнопок
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Заголовок
        header_label = ttk.Label(main_frame, text="Доступные индексы:", font=('Arial', 10, 'bold'))
        header_label.pack(pady=5)

        # Создаем фрейм с прокруткой
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Используем Treeview вместо Listbox для лучшего визуального представления
        tree = ttk.Treeview(list_frame, selectmode="browse", show="tree",
                            yscrollcommand=scrollbar.set)
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=tree.yview)

        # Информационная панель
        info_frame = ttk.LabelFrame(main_frame, text="Информация об индексе")
        info_frame.pack(fill=tk.X, pady=5)

        info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        info_text.pack(fill=tk.X, padx=5, pady=5)
        info_text.config(state=tk.DISABLED)

        def update_index_list():
            """Обновление списка индексов"""
            tree.delete(*tree.get_children())
            current_indices = list(es.indices.get_alias().keys())
            for idx in current_indices:
                tree.insert("", tk.END, text=idx, values=(idx,))

        def show_index_info(event):
            """Показать информацию о выбранном индексе"""
            selection = tree.selection()
            if selection:
                selected_index = tree.item(selection[0])['text']
                try:
                    # Получаем статистику индекса
                    stats = es.indices.stats(index=selected_index)
                    info = (f"Документов: {stats['indices'][selected_index]['total']['docs']['count']}\n"
                            f"Размер: {stats['indices'][selected_index]['total']['store']['size_in_bytes'] / 1024:.2f} KB\n"
                            f"Количество полей: {len(es.indices.get_mapping(index=selected_index)[selected_index]['mappings'].get('properties', {}))}")

                    info_text.config(state=tk.NORMAL)
                    info_text.delete(1.0, tk.END)
                    info_text.insert(tk.END, info)
                    info_text.config(state=tk.DISABLED)
                except Exception as e:
                    logging.error(f"Error getting index info: {e}")

        tree.bind('<<TreeviewSelect>>', show_index_info)

        def on_select():
            """Обработчик выбора индекса"""
            selection = tree.selection()
            if selection:
                global index_name
                index_name = tree.item(selection[0])['text']
                dialog.destroy()
                messagebox.showinfo("Информация", f"Выбран индекс: {index_name}")
                perform_search()  # Обновляем результаты поиска для нового индекса

        def on_delete():
            """Обработчик удаления индекса"""
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("Предупреждение", "Выберите индекс для удаления")
                return

            selected_index = tree.item(selection[0])['text']
            if selected_index == index_name:
                messagebox.showwarning("Предупреждение",
                                       "Нельзя удалить текущий активный индекс. "
                                       "Сначала выберите другой индекс.")
                return

            if messagebox.askyesno("Подтверждение",
                                   f"Вы уверены, что хотите удалить индекс {selected_index}?\n"
                                   "Это действие необратимо!"):
                try:
                    es.indices.delete(index=selected_index)
                    messagebox.showinfo("Успех", f"Индекс {selected_index} успешно удален")
                    update_index_list()  # Обновляем список индексов

                    # Очищаем информационную панель
                    info_text.config(state=tk.NORMAL)
                    info_text.delete(1.0, tk.END)
                    info_text.config(state=tk.DISABLED)
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка при удалении индекса: {str(e)}")
                    logging.error(f"Index deletion error: {str(e)}")

        # Frame для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Кнопки действий
        ttk.Button(button_frame, text="Выбрать",
                   command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Удалить",
                   command=on_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Обновить список",
                   command=update_index_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Закрыть",
                   command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # Заполняем список индексов
        update_index_list()

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка получения списка индексов: {str(e)}")
        logging.error(f"Index selection error: {str(e)}")



def search_index(query):
    """Улучшенный поиск с AND логикой для частичных совпадений слов"""
    try:
        if not query:
            return []

        # Разбиваем запрос на отдельные слова
        query_terms = query.strip().split()

        search_body = {
            "query": {
                "bool": {
                    "must": [
                        # Каждое слово должно присутствовать (логическое И)
                        {
                            "multi_match": {
                                "query": term,
                                "fields": ["*"],
                                "type": "phrase_prefix",
                                "operator": "or"
                            }
                        } for term in query_terms
                    ]
                }
            },
            "size": 100
        }

        response = es.search(
            index=index_name,
            body=search_body
        )
        hits = response["hits"]["hits"]
        logging.info(f"Найдено {len(hits)} записей по запросу: {query}")
        return hits
    except Exception as e:
        logging.error(f"Ошибка поиска: {e}")
        messagebox.showerror("Ошибка", f"Ошибка при поиске: {str(e)}")
        return []

def update_table(data):
    """Обновление таблицы результатов"""
    # Очистка существующих данных
    for row in tree.get_children():
        tree.delete(row)

    if not data:
        return

    # Получаем все уникальные ключи из результатов
    all_columns = set()
    for result in data:
        all_columns.update(result["_source"].keys())

    # Обновляем колонки таблицы
    global columns
    columns = list(all_columns)
    tree["columns"] = columns

    # Настраиваем заголовки
    for col in columns:
        tree.heading(col, text=col.title())
        tree.column(col, width=100)

    # Добавляем данные
    for result in data:
        values = [result["_source"].get(col, "") for col in columns]
        tree.insert("", "end", values=values)


def perform_search(*args):
    """Выполнение поиска с задержкой"""
    query = search_entry.get()
    if len(query) >= 3:
        results = search_index(query)
        update_table(results)


def clear_search():
    """Очистка поиска"""
    search_entry.delete(0, tk.END)
    update_table([])


def copy_selected():
    """Копирование выбранной ячейки в буфер обмена"""
    selection = tree.selection()
    if not selection:
        return

    item = tree.item(selection[0])
    column = tree.identify_column(tree.winfo_pointerx() - tree.winfo_rootx())
    col_num = int(column.replace('#', '')) - 1

    try:
        value = item['values'][col_num]
        root.clipboard_clear()
        root.clipboard_append(str(value))
        status_text.config(text="Скопировано в буфер обмена")
        root.after(3000, lambda: status_text.config(text="Готово"))
    except Exception as e:
        logging.error(f"Copy error: {str(e)}")


def show_about():
    """Показать информацию о программе"""
    about_text = """
    Elasticsearch Search UI
    Версия 1.0

    Программа для работы с Elasticsearch:
    - Импорт данных из CSV
    - Поиск по индексам
    - Экспорт результатов в XLSX

    Поддерживаемые функции:
    - Полнотекстовый поиск
    - Импорт CSV с различными кодировками и разделителями
    - Выбор индекса для поиска
    - Экспорт результатов
    """
    messagebox.showinfo("О программе", about_text)


# Создание главного окна
root = tk.Tk()
root.title("Elasticsearch Search UI")

# Настройка стилей
style = ttk.Style()
style.configure("Treeview", rowheight=25)

# Создание меню
menubar = tk.Menu(root)
root.config(menu=menubar)

# Меню "Файл"
file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Файл", menu=file_menu)
file_menu.add_command(label="Импорт CSV", command=import_csv_dialog)
file_menu.add_command(label="Экспорт в XLSX", command=export_to_xlsx)
file_menu.add_separator()
file_menu.add_command(label="Выход", command=root.quit)

# Меню "Индекс"
index_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Индекс", menu=index_menu)
index_menu.add_command(label="Выбрать индекс", command=select_index)
index_menu.add_separator()
index_menu.add_command(label="Очистить поиск", command=clear_search)

# Меню "Помощь"
help_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Помощь", menu=help_menu)
help_menu.add_command(label="О программе", command=show_about)


# Создание основного интерфейса
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Поле поиска
search_frame = ttk.Frame(main_frame)
search_frame.pack(fill=tk.X, padx=10, pady=5)

search_label = ttk.Label(search_frame, text="Поиск:")
search_label.pack(side=tk.LEFT, padx=5)

search_entry = ttk.Entry(search_frame, width=50)
search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
search_entry.bind("<KeyRelease>", perform_search)

# Таблица результатов
table_frame = ttk.Frame(main_frame)
table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Инициализация пустого списка колонок
columns = []

# Создание таблицы с прокруткой
tree = ttk.Treeview(table_frame, columns=columns, show="headings")
scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=tree.xview)

tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_x.pack(fill=tk.X, padx=10)

# Контекстное меню для таблицы
popup_menu = tk.Menu(root, tearoff=0)
popup_menu.add_command(label="Копировать", command=copy_selected)

def show_popup(event):
    """Показать контекстное меню"""
    if tree.identify_row(event.y):  # Проверяем, что клик был по строке
        popup_menu.post(event.x_root, event.y_root)

tree.bind("<Button-3>", show_popup)  # Правый клик мыши

# Кнопки управления
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, padx=10, pady=5)

ttk.Button(button_frame, text="Импорт CSV",
          command=import_csv_dialog).pack(side=tk.LEFT, padx=5)

ttk.Button(button_frame, text="Экспорт в XLSX",
          command=export_to_xlsx).pack(side=tk.LEFT, padx=5)

ttk.Button(button_frame, text="Выбор индекса",
          command=select_index).pack(side=tk.LEFT, padx=5)

ttk.Button(button_frame, text="Очистить поиск",
          command=clear_search).pack(side=tk.LEFT, padx=5)

# Строка состояния
status_frame = ttk.Frame(root)
status_frame.pack(fill=tk.X, side=tk.BOTTOM)

status_text = ttk.Label(status_frame, text="Готово")
status_text.pack(side=tk.LEFT, padx=5)

# Статус Elasticsearch
es_status_label = ttk.Label(status_frame, text="")
es_status_label.pack(side=tk.RIGHT, padx=5)

def update_es_status():
    """Обновление статуса подключения к Elasticsearch"""
    if check_elasticsearch_health():
        es_status_label.config(text="Elasticsearch: Connected", foreground="green")
    else:
        es_status_label.config(text="Elasticsearch: Disconnected", foreground="red")
    root.after(5000, update_es_status)  # Обновление каждые 5 секунд

# Горячие клавиши
root.bind('<Control-f>', lambda e: search_entry.focus())
root.bind('<Control-l>', lambda e: clear_search())
root.bind('<Control-i>', lambda e: import_csv_dialog())
root.bind('<Control-e>', lambda e: export_to_xlsx())

# Обработчик закрытия окна
def on_closing():
    """Обработка закрытия приложения"""
    if messagebox.askokcancel("Выход", "Вы действительно хотите выйти?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Запуск обновления статуса и установка размера окна
update_es_status()
root.geometry("1200x800")

# Запуск главного цикла
if __name__ == "__main__":
    try:
        if not check_elasticsearch_health():
            messagebox.showwarning(
                "Предупреждение",
                "Не удалось подключиться к Elasticsearch. "
                "Проверьте, запущен ли сервер."
            )
        root.mainloop()
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}")
        messagebox.showerror("Критическая ошибка", str(e))


