import os
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from src.get_data import get_crypto_data
import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mdates

class CryptoApp:
    def __init__(self, window):
        self.window = window
        self.window.geometry("1920x1080")  # 16:9
        self.window.title("Finance Analysis")
        self.current_symbol_data = {}
        self.current_display_data = None

        # Ustalenie pełnej ścieżki do plików TCL
        current_dir = os.path.dirname(os.path.realpath(__file__))
        forest_light_path = os.path.join(current_dir, "src", "themes", "forest-light.tcl")
        forest_dark_path = os.path.join(current_dir, "src", "themes", "forest-dark.tcl")

        # Dodanie stylu
        style = ttk.Style(self.window)
        self.window.tk.call("source", forest_light_path)
        self.window.tk.call("source", forest_dark_path)
        style.theme_use("forest-dark")

        self.window.grid_columnconfigure(0, minsize=330)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        self.frame1 = ttk.Frame(self.window, style="TFrame", height=self.window.winfo_height())
        self.frame1.grid(row=0, column=0, sticky=tk.NSEW)
        self.frame1.grid_columnconfigure(0, weight=1)
        self.frame1.grid_propagate(False)

        self.widgets_frame = ttk.LabelFrame(self.frame1, width=1, text="Set data")
        self.widgets_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.widgets_frame.grid_columnconfigure(0, weight=1)

        self.symbol_list_label = ttk.Label(self.widgets_frame, text="Symbol:")
        self.symbol_list_label.grid(row=0, column=0, padx=15, pady=5, sticky="nsew")

        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Combobox(self.widgets_frame, textvariable=self.symbol_var)
        self.symbol_entry['values'] = ["BTC/USD", "ETH/USD", "LTC/USD"]
        self.symbol_entry.current(0)
        self.symbol_entry.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")
        self.symbol_entry.bind("<<ComboboxSelected>>", self.load_data)

        self.option_frame = ttk.LabelFrame(self.frame1, width=1, text="Set options")
        self.option_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.option_frame.grid_columnconfigure(0, weight=1)

        self.period_label = tk.Label(self.option_frame, text="Okres:")
        self.period_label.grid(row=0, column=0, padx=20, pady=10)

        self.period_var = tk.StringVar()
        self.period_entry = ttk.Combobox(self.option_frame, textvariable=self.period_var)
        self.period_entry['values'] = ["Rok", "Miesiąc", "Tydzień", "Dzień"]
        self.period_entry.current(0)
        self.period_entry.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")
        self.period_entry.bind("<<ComboboxSelected>>", self.update_chart)

        self.chart_type_label = tk.Label(self.option_frame, text="Typ wykresu:")
        self.chart_type_label.grid(row=2, column=0, padx=20, pady=10)

        self.chart_type_var = tk.StringVar()
        self.chart_type_line = ttk.Checkbutton(self.option_frame, text="Liniowy", variable=self.chart_type_var,
                                               onvalue="Liniowy", offvalue="")
        self.chart_type_line.grid(row=3, column=0, padx=15, pady=5, sticky="nsew")
        self.chart_type_candle = ttk.Checkbutton(self.option_frame, text="Świeczki", variable=self.chart_type_var,
                                                 onvalue="Świeczki", offvalue="")
        self.chart_type_candle.grid(row=4, column=0, padx=15, pady=5, sticky="nsew")
        self.chart_type_line.invoke()  # Set default to line chart
        self.chart_type_var.trace_add('write', self.update_chart)

        self.frame2 = ttk.Frame(self.window, height=self.window.winfo_height(), style="Green.TFrame")
        self.frame2.grid(row=0, column=1, sticky=tk.NSEW)
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(0, weight=3)
        self.frame_chart = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_chart.grid(row=0, column=0, sticky=tk.NSEW)
        self.frame_chart.grid_propagate(False)
        self.frame2.grid_rowconfigure(1, weight=2)
        self.frame_analise = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_analise.grid(row=1, column=0, sticky=tk.NSEW)
        self.frame_analise.grid_propagate(False)

        self.load_data()

    def load_data(self, event=None):
        symbol = self.symbol_var.get()
        if symbol in self.current_symbol_data:
            self.update_chart()
            return
        end_date = datetime.now().replace(tzinfo=None)
        start_date = end_date - timedelta(days=365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        timeframe = TimeFrame.Hour

        data = get_crypto_data(symbol, timeframe, start_date_str, end_date_str)
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)  # Convert to timezone-naive
        self.current_symbol_data[symbol] = data
        self.update_chart()

    def update_chart(self, *args):
        symbol = self.symbol_var.get()
        period = self.period_var.get()

        data = self.current_symbol_data[symbol]
        end_date = datetime.now().replace(tzinfo=None)
        if period == "Rok":
            start_date = end_date - timedelta(days=365)
        elif period == "Miesiąc":
            start_date = end_date - timedelta(days=30)
        elif period == "Tydzień":
            start_date = end_date - timedelta(days=7)
        elif period == "Dzień":
            start_date = end_date - timedelta(days=1)
        self.current_display_data = data[data['timestamp'] >= start_date]

        self.plot_crypto_data(self.current_display_data)

    def plot_crypto_data(self, data):
        for child in self.frame_chart.winfo_children():
            child.destroy()

        container_frame = ttk.Frame(self.frame_chart)
        container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#313131')
        fig.patch.set_alpha(1.0)
        ax.patch.set_facecolor('#313131')
        ax.patch.set_alpha(0.2)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        if self.chart_type_var.get() == "Liniowy":
            ax.plot(data['timestamp'], data['close'], label=self.symbol_var.get())
        else:
            mpf.plot(data.set_index('timestamp'), type='candle', ax=ax, style='charles', show_nontrading=True, warn_too_much_data=100000)

        ax.set_xlabel('Data')
        ax.set_ylabel('Cena zamknięcia')
        ax.set_title(f'Cena zamknięcia {self.symbol_var.get()}')
        ax.legend()
        ax.grid(True)

        self.adjust_xaxis_labels(ax, data)

        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, container_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.display_analyse(data)

    def adjust_xaxis_labels(self, ax, data):
        num_days = (data['timestamp'].max() - data['timestamp'].min()).days
        if num_days > 365:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif num_days > 30:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif num_days > 7:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def display_analyse(self, data):
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        columns = ["Time unit", "Price change", "Open price", "Close price", "Prediction 1 algorithm",
                   "Prediction 2 algorithm", "Prediction 3 algorithm", "Prediction 4 algorithm",
                   'Average prediction', 'Rows analysed']

        tree = ttk.Treeview(self.frame_analise, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")

        predictions = self.make_predictions(data)
        avg_predictions = pd.DataFrame(predictions).mean(axis=1)

        time_units = {
            "Rok": 365,
            "Miesiąc": 30,
            "Tydzień": 7,
            "Dzień": 1,
            "Godzina": 1 / 24
        }

        for unit, days in time_units.items():
            unit_data = data[data['timestamp'] >= (data['timestamp'].max() - pd.Timedelta(days=days))]
            price_change = (unit_data['close'].iloc[-1] - unit_data['close'].iloc[0]) / unit_data['close'].iloc[0] * 100
            avg_prediction = avg_predictions.mean()
            row = [unit, f"{price_change:.2f}%", unit_data['open'].iloc[0], unit_data['close'].iloc[-1],
                   predictions["Prediction 1 algorithm"][-1], predictions["Prediction 2 algorithm"][-1],
                   predictions["Prediction 3 algorithm"][-1], predictions["Prediction 4 algorithm"][-1],
                   avg_prediction, len(unit_data)]
            tree.insert("", "end", values=row)

        tree.pack(side="left", fill="both", expand=True)

    @staticmethod
    def make_predictions(data):
        return {
            "Prediction 1 algorithm": [data['close'].mean()] * len(data),
            "Prediction 2 algorithm": [data['close'].mean()] * len(data),
            "Prediction 3 algorithm": [data['close'].mean()] * len(data),
            "Prediction 4 algorithm": [data['close'].mean()] * len(data)
        }

if __name__ == "__main__":
    window = tk.Tk()
    app = CryptoApp(window)
    window.mainloop()
