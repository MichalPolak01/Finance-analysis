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
import numpy as np

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
        self.period_entry['values'] = ["Rok", "Pół Roku", "Miesiąc", "Dwa Tygodnie", "Tydzień", "Trzy Dni", "Dzień", "12h", "6h", "30min", "15min"]
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

        self.fib_var = tk.BooleanVar()
        self.fib_check = ttk.Checkbutton(self.option_frame, text="Rysuj Fibonacci", variable=self.fib_var, command=self.update_chart)
        self.fib_check.grid(row=5, column=0, padx=15, pady=5, sticky="nsew")

        self.zigzag_var = tk.BooleanVar()
        self.zigzag_check = ttk.Checkbutton(self.option_frame, text="Rysuj ZigZag", variable=self.zigzag_var, command=self.update_chart)
        self.zigzag_check.grid(row=6, column=0, padx=15, pady=5, sticky="nsew")

        self.lwma_var = tk.BooleanVar()
        self.lwma_check = ttk.Checkbutton(self.option_frame, text="Rysuj LWMA", variable=self.lwma_var, command=self.update_chart)
        self.lwma_check.grid(row=7, column=0, padx=15, pady=5, sticky="nsew")

        self.volume_var = tk.BooleanVar()
        self.volume_check = ttk.Checkbutton(self.option_frame, text="Pokaż wolumen", variable=self.volume_var, command=self.update_chart)
        self.volume_check.grid(row=8, column=0, padx=15, pady=5, sticky="nsew")

        self.frame2 = ttk.Frame(self.window, height=self.window.winfo_height(), style="Green.TFrame")
        self.frame2.grid(row=0, column=1, sticky=tk.NSEW)
        self.frame2.grid_columnconfigure(0, weight=1)
        self.frame2.grid_rowconfigure(0, weight=3)
        self.frame_chart = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_chart.grid(row=0, column=0, sticky=tk.NSEW)
        self.frame_chart.grid_propagate(False)
        self.frame2.grid_rowconfigure(1, weight=1)
        self.frame_volume = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_volume.grid(row=1, column=0, sticky=tk.NSEW)
        self.frame_volume.grid_propagate(False)
        self.frame2.grid_rowconfigure(2, weight=2)
        self.frame_analise = ttk.Frame(self.frame2, style="TNotebook", padding=10)
        self.frame_analise.grid(row=2, column=0, sticky=tk.NSEW)
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
        elif period == "Pół Roku":
            start_date = end_date - timedelta(days=182)
        elif period == "Miesiąc":
            start_date = end_date - timedelta(days=30)
        elif period == "Dwa Tygodnie":
            start_date = end_date - timedelta(days=14)
        elif period == "Tydzień":
            start_date = end_date - timedelta(days=7)
        elif period == "Trzy Dni":
            start_date = end_date - timedelta(days=3)
        elif period == "Dzień":
            start_date = end_date - timedelta(days=1)
        elif period == "12h":
            start_date = end_date - timedelta(hours=12)
        elif period == "6h":
            start_date = end_date - timedelta(hours=6)
        elif period == "30min":
            start_date = end_date - timedelta(minutes=30)
        elif period == "15min":
            start_date = end_date - timedelta(minutes=15)
        else:
            start_date = end_date - timedelta(days=365)
        data = data[data['timestamp'] >= start_date]
        self.current_display_data = data

        # Clear previous charts
        for widget in self.frame_chart.winfo_children():
            widget.destroy()
        for widget in self.frame_volume.winfo_children():
            widget.destroy()
        for widget in self.frame_analise.winfo_children():
            widget.destroy()

        self.draw_chart()
        if self.volume_var.get():
            self.draw_volume()
        self.display_analyse(data)

    def draw_chart(self):
        plt.close('all')  # Zamknij wszystkie otwarte figury

        data = self.current_display_data

        print("Drawing chart for data range:")
        print(data['timestamp'].min(), "to", data['timestamp'].max())

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#313131')
        ax.set_facecolor('#313131')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')  # Ustawienie koloru etykiet osi Y na biały
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("white")

        if self.chart_type_var.get() == "Liniowy":
            ax.plot(data['timestamp'], data['close'], color='blue')
        else:
            mpf.plot(data.set_index('timestamp'),
                     type='candle',
                     ax=ax,
                     style='charles',
                     show_nontrading=True,
                     warn_too_much_data=100000)

        print("Calling adjust_xaxis_labels")
        self.adjust_xaxis_labels(ax, data)

        print("Adjusting x-axis labels")
        # Determine the time range in the data
        time_diff = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]

        if time_diff.days > 365:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif time_diff.days > 30:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif time_diff.days > 7:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Ustawienie koloru etykiet osi Y na biały
        ax.tick_params(axis='y', colors='white')

        # Draw Fibonacci retracement if checked
        if self.fib_var.get():
            self.draw_fibonacci(ax)

        # Draw ZigZag if checked
        if self.zigzag_var.get():
            self.draw_zigzag(ax, data)

        # Draw LWMA if checked
        if self.lwma_var.get():
            self.draw_lwma(ax, data)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_chart)
        toolbar = NavigationToolbar2Tk(canvas, self.frame_chart)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        print("Chart drawn.")


    def draw_volume(self):
        data = self.current_display_data

        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor('#313131')
        ax.set_facecolor('#313131')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("white")

        ax.bar(data['timestamp'], data['volume'], color='gray')
        self.adjust_xaxis_labels(ax, data)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_volume)
        toolbar = NavigationToolbar2Tk(canvas, self.frame_volume)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()


        if period == "Rok":
            start_date = end_date - timedelta(days=365)
        elif period == "Pół Roku":
            start_date = end_date - timedelta(days=182)
        elif period == "Miesiąc":
            start_date = end_date - timedelta(days=30)
        elif period == "Dwa Tygodnie":
            start_date = end_date - timedelta(days=14)
        elif period == "Tydzień":
            start_date = end_date - timedelta(days=7)
        elif period == "Trzy Dni":
            start_date = end_date - timedelta(days=3)
        elif period == "Dzień":
            start_date = end_date - timedelta(days=1)
        elif period == "12h":
            start_date = end_date - timedelta(hours=12)
        elif period == "6h":
            start_date = end_date - timedelta(hours=6)
        elif period == "30min":
            start_date = end_date - timedelta(minutes=30)
        elif period == "15min":
            start_date = end_date - timedelta(minutes=15)
        else:
            start_date = end_date - timedelta(days=365)
        data = data[data['timestamp'] >= start_date]
        self.current_display_data = data

    def adjust_xaxis_labels(self, ax, data):
        num_days = (data['timestamp'].max() - data['timestamp'].min()).days
        if num_days > 365:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif num_days > 180:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=(1, 15)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        elif num_days > 30:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        elif num_days > 7:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        elif num_days > 3:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
        elif num_days > 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif num_days > 0.5:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif num_days > 0.25:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif (data['timestamp'].max() - data['timestamp'].min()).seconds > 3600:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.MinuteLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def draw_fibonacci(self, ax):
        data = self.current_display_data
        price_min = data['low'].min()
        price_max = data['high'].max()
        diff = price_max - price_min

        levels = [price_max - diff * ratio for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]

        for level in levels:
            ax.axhline(level, linestyle='--', alpha=0.5)
            ax.text(data['timestamp'].iloc[-1], level, f'{level:.2f}', alpha=0.5, color='green')

    def draw_zigzag(self, ax, data):
        if len(data) < 0:
            # Not enough data to perform ZigZag calculation
            return

        peak_valley = self.peak_valley_pivots(data['close'], 0.02, -0.02)
        pivots = data.loc[peak_valley != 0]
        if not pivots.empty:
            ax.plot(pivots['timestamp'], pivots['close'], color='yellow', label='ZigZag', marker='o', linestyle='-',
                    markersize=5)
            ax.legend()
            self.adjust_xaxis_labels(ax, data)  # Adjust x-axis labels for correct date formatting

    def peak_valley_pivots(self, close, up_thresh, down_thresh):
        if len(close) < 2:
            return np.zeros(len(close))  # Return an array of zeros if not enough data

        pivots = np.zeros(len(close))
        up = down = None

        for i in range(1, len(close)):
            if up is None or close.iloc[i] >= close.iloc[up]:
                up = i
            if down is None or close.iloc[i] <= close.iloc[down]:
                down = i

            if up is not None and close.iloc[up] - close.iloc[i] > close.iloc[up] * down_thresh:
                pivots[up] = 1
                down = up = None
            elif down is not None and close.iloc[i] - close.iloc[down] > close.iloc[down] * up_thresh:
                pivots[down] = -1
                down = up = None

        return pivots

    def draw_lwma(self, ax, data):
        period = 1  # Example LWMA period
        if len(data) < period:
            # Not enough data to calculate LWMA
            return

        weights = np.arange(1, period + 1)
        lwma = data['close'].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        ax.plot(data['timestamp'], lwma, color='red', label='LWMA')
        ax.legend()
        self.adjust_xaxis_labels(ax, data)

    def display_analyse(self, data):
        columns = ["Time unit", "Price change (%)", "Open price", "Close price", "Prediction 1 algorithm",
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
            "Pół Roku": 182,
            "Miesiąc": 30,
            "Dwa Tygodnie": 14,
            "Tydzień": 7,
            "Trzy Dni": 3,
            "Dzień": 1,
            "12h": 0.5,
            "6h": 0.25,
            "30min": 1 / 48,
            "15min": 1 / 96
        }

        for unit, days in time_units.items():
            unit_data = data[data['timestamp'] >= (data['timestamp'].max() - pd.Timedelta(days=days))]
            if len(unit_data) > 0:
                price_change = (unit_data['close'].iloc[-1] - unit_data['close'].iloc[0]) / unit_data['close'].iloc[0] * 100
                avg_prediction = avg_predictions.mean()
                row = [unit, f"{price_change:.2f}", f"{unit_data['open'].iloc[0]:.2f}", f"{unit_data['close'].iloc[-1]:.2f}",
                       f"{predictions['Prediction 1 algorithm'][-1]:.2f}", f"{predictions['Prediction 2 algorithm'][-1]:.2f}",
                       f"{predictions['Prediction 3 algorithm'][-1]:.2f}", f"{predictions['Prediction 4 algorithm'][-1]:.2f}",
                       f"{avg_prediction:.2f}", len(unit_data)]
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
