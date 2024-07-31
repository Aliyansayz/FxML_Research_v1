# for Daily so far
class indicators:

  def ema(self, price, period):

    price = np.array(price)
    alpha = 2 / (period + 1.0)
    alpha_reverse = 1 - alpha
    data_length = len(price)

    power_factors = alpha_reverse ** (np.arange(data_length + 1))
    initial_offset = price[0] * power_factors[1:]

    scale_factors = 1 / power_factors[:-1]

    weight_factor = alpha * alpha_reverse ** (data_length - 1)

    weighted_price_data = price * weight_factor * scale_factors
    cumulative_sums = weighted_price_data.cumsum()
    ema_values = initial_offset + cumulative_sums * scale_factors[::-1]

    return ema_values


  def moving_min(self, array, period):
      moving_min = np.empty_like(array)
      moving_min = np.full(moving_min.shape, np.nan)
      for i in range(period, len(array)):
          moving_min[i] = np.min(array[i - period:i])

      # to be changed
      moving_min[np.isnan(moving_min)] = np.nanmean(moving_min)
      return moving_min

  def moving_max(self, array, period):
      moving_max = np.empty_like(array)
      moving_max = np.full(moving_max.shape, np.nan)
      # moving_max[:period] = np.max(array[:period])
      for i in range(period, len(array)):
          moving_max[i] = np.max(array[i - period:i])
      # to be changed
      moving_max[np.isnan(moving_max)] = np.nanmean(moving_max)
      return moving_max

  def true_range(self, high, low, close):

    close_shift = self.shift(close, 1)
    high_low, high_close, low_close = np.array(high - low, dtype=np.float32), \
                                      np.array(abs(high - close_shift), dtype=np.float32), \
                                      np.array(abs(low - close_shift), dtype=np.float32)

    true_range = np.max(np.hstack((high_low, high_close, low_close)).reshape(-1, 3), axis=1)

    return true_range


  def shift(self, array, place):

    array = np.array(array, dtype=np.float32)
    shifted = np.roll(array, place)
    shifted[0:place] = np.nan
    shifted[np.isnan(shifted)] = np.nanmean(shifted)

    return shifted


  def ma_based_supertrend_indicator(self, high, low, close, atr_length=10, atr_multiplier=3, ma_length=10):

      # Calculate True Range and Smoothed ATR
      tr  = self.true_range(high, low, close)
      atr = self.ema(tr, atr_length)

      upper_band = (high + low) / 2 + (atr_multiplier * atr)
      lower_band = (high + low) / 2 - (atr_multiplier * atr)

      trend = np.zeros(len(atr))

      # Calculate Moving Average
      ema_values = ema(close, ma_length)

      if ema_values[0] > lower_band[0]:
          trend[0] = lower_band[0]
      elif ema_values[0] < upper_band[0]:
          trend[0] = upper_band[0]
      else:
          trend[0] = upper_band[0]

      # Compute final upper and lower bands
      for i in range(1, len(close)):
          if ema_values[i] > trend[i - 1]:
              trend[i] = max(trend[i - 1], lower_band[i])


          elif ema_values[i] < trend[i - 1]:
              trend[i] = min(trend[i - 1], upper_band[i])

          else:
              trend[i] = trend[i - 1]

      status_value = np.where(ema_values > trend, 1.0, -1.0)

      return trend, status_value




  def supertrend_status_crossover(self, status_value):


      prev_status = np.roll(status_value, 1)
      supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0, np.where((prev_status > 0) & (status_value < 0), -1.0, 0))

      return supertrend_status_crossover




  def supertrend_indicator(self, high, low, close, period, multiplier=1.0):

      true_range_value = self.true_range(high, low, close)

      smoothed_atr = self.ema(true_range_value, period)

      upper_band = (high + low) / 2 + (multiplier * smoothed_atr)
      lower_band = (high + low) / 2 - (multiplier * smoothed_atr)

      supertrend = np.zeros(len(true_range_value))
      trend = np.zeros(len(true_range_value))

      if close[0] > upper_band[0]: trend[0] = upper_band[0]
      elif close[0] < lower_band[0]: trend[0] = lower_band[0]
      else:  trend[0] = upper_band[0]

      for i in range(1, len(close)):

          if close[i] > upper_band[i]: trend[i] = upper_band[i]
          elif close[i] < lower_band[i]: trend[i] = lower_band[i]
          else: trend[i] = trend[i - 1]

      # Calculate Buy/Sell Signals using numpy where  # np.where( close > trend, '1 Buy', '-1 Sell')
      status_value = np.where(close > trend, 1.0, -1.0)

      return trend, status_value

  def supertrend_status_crossover(self, status_value):


      prev_status = np.roll(status_value, 1)
      supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0, np.where((prev_status > 0) & (status_value < 0), -1.0, 0))

      return supertrend_status_crossover




  def direction_crossover_signal_line(self, signal, signal_ema):

      direction = np.where(signal - signal_ema > 0, 1, -1)
      prev_direction = np.roll(direction, 1)
      crossover = np.where((prev_direction < 0) & (direction > 0), 1,
                            np.where((prev_direction > 0) & (direction < 0), -1, 0))

      return direction, crossover


  def stochastic_momentum_index(self, high, low, close, period=20, ema_period=5):

      lengthD = ema_period
      lowest_low   = self.moving_min(low, period)
      highest_high = self.moving_max(high, period)
      relative_range = close - ((highest_high + lowest_low) / 2)
      highest_lowest_range = highest_high - lowest_low

      relative_range_smoothed = self.ema(self.ema(relative_range, ema_period), ema_period)
      highest_lowest_range_smoothed = self.ema(self.ema(highest_lowest_range, ema_period), ema_period)

      smi = [(relative_range_smoothed[i] / (highest_lowest_range_smoothed[i] / 2)) * 100 if
              highest_lowest_range_smoothed[i] != 0 else 0.0
              for i in range(len(relative_range_smoothed))]

      smi_ema = self.ema(smi, ema_period)

      return smi, smi_ema


  def candle_type(self, o, h, l, c):

      diff = abs(c - o)
      o1, c1 = np.roll(o, 1), np.roll(c, 1)  #
      min_oc = np.where(o < c, o, c)
      max_oc = np.where(o > c, o, c)

      pattern = np.where(
        np.logical_and( min_oc - l > diff, h - max_oc < diff), 6,
        np.where(np.logical_and( h - max_oc > diff, min_oc - l < diff),
        4, np.where(np.logical_and(np.logical_and(c > o, c1 < o1), np.logical_and(c > o1, o < c1)),
          5, np.where( min_oc - l > diff, 3,
                        np.where(np.logical_and( h - max_oc > diff,
                    min_oc - l < diff),
                        2, np.where(np.logical_and(np.logical_and(c > o, c1 < o1), np.logical_and(c > o1, o < c1)),
                        1, 0))))))
      return pattern





  def heikin_ashi_status(self,  ha_open, ha_close):

      candles = np.full_like(ha_close, '', dtype='U10')

      for i in range(1, len(ha_close)):

          if ha_close[i] > ha_open[i]: candles[i] = 2 #'Green'

          elif ha_close[i] < ha_open[i]: candles[i] = 1 # 'Red'

          else: candles[i] = 0 #'Neutral'

      return candles

  def heikin_ashi_candles(self, open, high, low, close):

      ha_low, ha_close = np.empty(len(close), dtype=np.float32), np.empty(len(close), dtype=np.float32)
      ha_open, ha_high = np.empty(len(close), dtype=np.float32), np.empty(len(close), dtype=np.float32)

      ha_open[0] = (open[0] + close[0]) / 2
      ha_close[0] = (close[0] + open[0] + high[0] + low[0]) / 4

      for i in range(1, len(close)):
          ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
          ha_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4
          ha_high[i] = max(high[i], ha_open[i], ha_close[i])
          ha_low[i] = min(low[i], ha_open[i], ha_close[i])

      return ha_open, ha_close, ha_high, ha_low


# =============================================================================================

from datetime import date, timedelta
from math import ceil



class currency_model(indicators):


  @classmethod
  def load_csv_file(cls, file_path ): # Load csv file or 

    data = pd.read_csv(file_path)


    return data


  @classmethod
  def get_data(cls, symbol, start_date, end_date): # Load market data from api 

    cls.symbol = symbol[:6]
    data = yf.download(f'{symbol}', start=start_date, end=end_date)

    return data

  @classmethod
  def get_data_for_today_prediction(cls, symbol, start_date, end_date):

    cls.symbol = symbol[:6]
    data = yf.download(f'{symbol}', start=start_date, end=end_date)  # end date will be of today.

    return data


  @classmethod
  def get_data_for_tomorrow_prediction(cls, symbol, start_date):

    cls.symbol = symbol[:6]
    data = yf.download(f'{symbol}', start=start_date ) # end date omitted.

    formatted_date = cls.get_current_date(tomorrow=True)
    new_date = pd.to_datetime(f'{formatted_date}') # adding tomorrow date if it's sat, sun then automatically monday selected.

    data.loc[new_date] = np.nan
    last_row_index = data.index[-1]

    day_of_week, week_of_month, month = cls.get_day_of_week_month_week(tomorrow=True)
    data.loc[last_row_index, 'Day_of_Week'] = day_of_week
    data.loc[last_row_index, 'Week_of_Month'] = week_of_month
    data.loc[last_row_index, 'Month'] = month

    return data

  @classmethod
  def get_current_date(cls, tomorrow=None ):
    from datetime import date, timedelta

    # Get today's date
    if tomorrow :
      today_date = date.today() + timedelta(days=1)
    else:
      today_date = date.today()

    if today_date.isoweekday() == 6:    today_date += timedelta(days=2) # Saturday
    elif today_date.isoweekday() == 7:  today_date += timedelta(days=1) # Sunday

    formatted_date = today_date.strftime('%Y-%m-%d')

    return formatted_date

  @classmethod
  def get_day_of_week_month_week(cls, tomorrow=None ):

      if tomorrow :
        today_date = date.today() + timedelta(days=1)
      else:
        today_date = date.today()

      if today_date.isoweekday() == 6:  # Saturday
          today_date += timedelta(days=2)
      elif today_date.isoweekday() == 7:  # Sunday
          today_date += timedelta(days=1)

      # print(formatted_date)

      day_of_week = today_date.isoweekday()

      first_day_of_month = today_date.replace(day=1)

      week_of_month = ceil((today_date.day + first_day_of_month.isoweekday() - 1) / 7)

      month = today_date.month

      return  day_of_week, week_of_month, month

  @classmethod
  def prepare_data(cls, data):
    indicator = cls()
    open, high, low, close = data['Open'] , data['High'], data['Low'], data['Close']

    elastic_supertrend, es_status_value = indicator.ma_based_supertrend_indicator( high, low, close, atr_length=10, atr_multiplier=2.5, ma_length=10)

    elastic_supertrend_crossover = indicator.supertrend_status_crossover(es_status_value)

    supertrend, supertrend_status_value = indicator.supertrend_indicator(high, low, close, period= 10, multiplier=0.66)
    supertrend_crossover = indicator.supertrend_status_crossover(supertrend_status_value)

    smi, smi_ema = indicator.stochastic_momentum_index(high, low, close, period=9, ema_period=3)

    smi_direction, smi_crossover = direction_crossover_signal_line(smi, smi_ema)

    smi_fast, smi_ema = indicator.stochastic_momentum_index(high, low, close, period=9, ema_period=2)

    smi_fast_direction, smi_fast_crossover = indicator.direction_crossover_signal_line(smi_fast, smi_ema)

    ha_open, ha_close, ha_high, ha_low = indicator.heikin_ashi_candles(open, high, low, close)
    heikin_ashi_candle = indicator.heikin_ashi_status(ha_open, ha_close)

    candle_type_value  = indicator.candle_type(open, high, low, close)

    ema_3  = indicator.ema( (high+low/2), 3 )
    ema_5  = indicator.ema( (high+low/2), 5 )
    ema_7  = indicator.ema( (high+low/2), 7 )
    ema_14 = indicator.ema( (high+low/2), 14 )

    # Add seasonality features
    data['Day_of_Week'] = data.index.dayofweek + 1  # Monday=1, ..., Friday=5
    data['Week_of_Month'] = (data.index.day - 1) // 7 + 1
    data['Month'] = data.index.month

    # Add numerical features
    data['Prev_Close'] = data['Close'].shift(1)
    data['Price_Range'] = data['High'] - data['Low']
    data['Median_Price'] = (data['High'] + data['Low']) / 2

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['RSI']   = calculate_rsi(data['Adj Close'])
    data['STDEV'] = data['Close'].rolling(window=14).std()
    data['Upper_Band'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
    data['smi_crossover'] = np.asarray(smi_crossover)
    data['smi_direction'] = np.asarray(smi_direction)
    data['smi_value'] =  np.asarray(smi)
    data['heikin_ashi'] = np.asarray(heikin_ashi_candle)
    data['supertrend']  = np.asarray(supertrend_status_value)
    data['supertrend_crossover'] = np.asarray(supertrend_crossover)

    data['elastic_supertrend'] = es_status_value
    data['elastic_supertrend_cross'] = elastic_supertrend_crossover

    data['candle_type'] = candle_type_value
    data['smi_fast']    = smi_fast
    data['smi_fast_direction']  = smi_fast_direction
    data['smi_fast_crossover']  = smi_fast_crossover

    data['ema_3']  = ema_3
    data['ema_5']  = ema_5
    data['ema_7']  = ema_7
    data['ema_14'] = ema_14

    return data

  @classmethod
  def gather_features(cls, data):

        # Select specific columns (for example, 'A' and 'C')
    selected_columns = data[['ema_3', 'ema_5', 'ema_7']]

    # Calculate the variance of the selected columns
    data['ema_variance'] = selected_columns.var()

    data['ema_mean'] = selected_columns.mean()

    selected_columns = data[['ema_mean', 'ema_14']]
    data['secondary_variance'] = selected_columns.var()
    data['ema_difference'] = data['ema_mean'] - data['ema_14']

    # Calculate daily returns and classify them
    data['daily_returns'] = data['Adj Close'] - data['Adj Close'].shift(1)


    def classify_daily_returns(pip_change):
        if pip_change > 0:
            return 'Pips Rise'
        else:
            return 'Pips Fall'

    data['Target'] = data['daily_returns'].apply(classify_daily_returns)

    window_size = 9

    for i in range(1, window_size + 1):
        data[f'Day_of_Week_T-{i}'] = data['Day_of_Week'].shift(i)
        data[f'Week_of_Month_T-{i}'] = data['Week_of_Month'].shift(i)
        data[f'Month_T-{i}'] = data['Month'].shift(i)
        data[f'Close_T-{i}'] = data['Close'].shift(i)
        data[f'High_T-{i}'] = data['High'].shift(i)
        data[f'Low_T-{i}']  = data['Low'].shift(i)
        data[f'RSI_T-{i}']  = data['RSI'].shift(i)
        data[f'Upper_Band_T-{i}'] = data['Upper_Band'].shift(i)
        data[f'Lower_Band_T-{i}'] = data['Lower_Band'].shift(i)
        data[f'smi_crossover_T-{i}'] = data['smi_crossover'].shift(i)
        data[f'smi_direction_T-{i}'] = data['smi_direction'].shift(i)
        data[f'smi_value_T-{i}'] = data['smi_value'].shift(i)
        data[f'heikin_ashi_T-{i}'] = data['heikin_ashi'].shift(i)
        data[f'supertrend_T-{i}'] = data['supertrend'].shift(i)
        data[f'supertrend_crossover_T-{i}'] = data['supertrend_crossover'].shift(i)
        data[f'elastic_supertrend_T-{i}']   = data['elastic_supertrend'].shift(i)
        data[f'elastic_supertrend_cross_T-{i}'] = data['elastic_supertrend_cross'].shift(i)
        data[f'candle_type_T-{i}'] = data['candle_type'].shift(i)
        data[f'smi_fast_T-{i}']    = data['smi_fast'].shift(i)
        data[f'smi_fast_direction_T-{i}']  = data['smi_fast_direction'].shift(i)
        data[f'smi_fast_crossover_T-{i}']  = data['smi_fast_crossover'].shift(i)
        data[f'ema_variance_T-{i}']        = data['ema_variance'].shift(i)
        data[f'secondary_variance_T-{i}']  = data['secondary_variance'].shift(i)
        data[f'ema_3_T-{i}']  = data['ema_3'].shift(i)
        data[f'ema_5_T-{i}']  = data['ema_5'].shift(i)
        data[f'ema_7_T-{i}']  = data['ema_7'].shift(i)
        data[f'ema_14_T-{i}'] = data['ema_14'].shift(i)
        data[f'ema_difference_T-{i}'] = data['ema_difference'].shift(i)

        columns_with_empty_strings = data.columns[data.isin(['']).any()].tolist()
        data[columns_with_empty_strings] = data[columns_with_empty_strings].replace('', np.nan )

        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=[object]).columns

        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

        for col in categorical_cols:
            mode_val  = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)

        return data


    @classmethod
    def normalize_features(cls, data):

      features = ['Day_of_Week', 'Week_of_Month', 'Month' ] + \
                  [f'Day_of_Week_T-{i}' for i in range(1, window_size + 1)] +\
                  [f'Week_of_Month_T-{i}' for i in range(1, window_size + 1)] +\
                  [f'Month_T-{i}' for i in range(1, window_size + 1)]  +\
                  [f'Close_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'High_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'Low_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'RSI_T-{i}' for i in range(1, window_size + 1)   ] + \
                  [f'Upper_Band_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'Lower_Band_T-{i}' for i in range(1, window_size + 1) ]  + \
                  [f'smi_crossover_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'smi_direction_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'smi_value_T-{i}' for i in range(1, window_size + 1) ]     + \
                  [f'heikin_ashi_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'supertrend_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'supertrend_crossover_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'elastic_supertrend_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'elastic_supertrend_cross_T-{i}' for i in range(1, window_size + 1) ] + \
                  [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'smi_fast_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'smi_fast_direction_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'smi_fast_crossover_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'ema_3_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'ema_5_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'ema_7_T-{i}' for i in range(1, window_size + 1) ] +\
                  [f'ema_14_T-{i}'for i in range(1, window_size + 1) ] +\
                  [f'ema_difference_T-{i}' for i in range(1, window_size + 1) ]

      X = data[features]
      y = data['Target']

      X.fillna(0.0, inplace=True)

      le = LabelEncoder()
      y_encoded = le.fit_transform(y)

      return X, y_encoded, le
