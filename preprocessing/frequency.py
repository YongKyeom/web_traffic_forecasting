import calendar
import locale
import pandas as pd
import statsmodels.api as sm
import numpy as np

from statsmodels.formula.api import ols
from scipy.signal import find_peaks

def fn_findfrequency(
        df: pd.DataFrame,
        pvalueCUT: float = 0.2,
        pvalueCUT_sig: float = 0.05
) -> (bool, dict):
    """
    시계열 데이터의 Frequency를 찾는 함수

    Returns:
        frequency_detected: auto_arima에 Frequency를 적용할 지 여부(True/False)
        print_dict: Frequency 적용시, auto_arima함수에 사용할 인자를 담은 Dictionary
    """

    ## Locale 설정
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

    x = df['View']
    # 데이터의 길이
    n = len(x)

    if n < 60:
        use_frequency = False
        kwargs = {'with_intercept': True}
        # specified as a string where ‘c’ indicates a constant
        # non-seasonal and has intercept

        return use_frequency, kwargs

    # 자기상관 함수 계산
    acf = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf = acf[n - 1:] / acf[n - 1]

    # 자기상관 함수에서 첫 번째 peak 찾기
    peaks, _ = find_peaks(acf)

    if (len(peaks) != 0):
        first_peak = peaks[0]
    else:
        first_peak = 1

    train_last_month_day = calendar.monthrange(df.iloc[-1]['Date'].year, df.iloc[-1]['Date'].month)[1]

    df['day_of_week'] = df['Date'].dt.strftime('%a')
    df['day'] = df['Date'].dt.day
    df['last_day'] = df['Date'].apply(lambda x: x.day == calendar.monthrange(x.year, x.month)[1]).astype(int)
    df['month_order'] = df['Date'].dt.month - min(df['Date'].dt.month - 1)

    df = pd.concat([df, pd.get_dummies(df['day_of_week'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['day'], prefix='DAY')], axis=1)
    df = df.drop('day_of_week', axis=1).drop('day', axis=1)

    WeekdayTag_names = ['Wed', 'Sat', 'Sun']
    EachdayTag_names = ['DAY_' + str(i) for i in (1, 2, 3, 4, 5)] + ['last_day']
    Tag_names = WeekdayTag_names + EachdayTag_names

    df = df[['Date', 'View', 'month_order'] + Tag_names]

    # 변수 초기화
    ITEMPattern = []
    # pvalues for week & month
    pvalue_sig_week = []
    pvalue_sig_month = []

    # weekday effect
    for i in range(len(WeekdayTag_names)):
        temp_tag = df[['View', WeekdayTag_names[i]]]
        temp_lm = ols('View' + '~' + temp_tag.columns[1], data=temp_tag).fit()
        temp_anova_pvalue = sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]] < pvalueCUT
        temp_anova_pvalue_sig = sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]] < pvalueCUT_sig

        if not pd.isna(temp_anova_pvalue):
            if temp_anova_pvalue:
                ITEMPattern.append(temp_tag.columns[1])

        if not pd.isna(temp_anova_pvalue_sig):
            if temp_anova_pvalue_sig:
                pvalue_sig_week.append(sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]])
                ITEMPattern.append('SigWeek')

    # Daily effect
    for i in range(len(EachdayTag_names)):
        if np.isin(EachdayTag_names[i], df.columns):
            temp_tag = df[['View', EachdayTag_names[i]]]
            temp_lm = ols('View' + '~' + temp_tag.columns[1], data=temp_tag).fit()
            temp_anova_pvalue = sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]] < pvalueCUT
            temp_anova_pvalue_sig = sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]] < pvalueCUT_sig

        if not pd.isna(temp_anova_pvalue):
            if temp_anova_pvalue:
                ITEMPattern.append(temp_tag.columns[1])

        if not pd.isna(temp_anova_pvalue_sig):
            if temp_anova_pvalue_sig:
                pvalue_sig_month.append(sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]])
                ITEMPattern.append('SigMonth')

    # linear month effect
    temp_tag = df[['View', 'month_order']]
    temp_lm = ols('View' + '~' + temp_tag.columns[1], data=temp_tag).fit()
    temp_anova_pvalue = sm.stats.anova_lm(temp_lm, type=2)['PR(>F)'][temp_tag.columns[1]] < pvalueCUT_sig
    if not pd.isna(temp_anova_pvalue):
        if temp_anova_pvalue:
            ITEMPattern.append('MonthTrend')

    weekly = np.isin(ITEMPattern, WeekdayTag_names).sum()
    daily = np.isin(ITEMPattern, EachdayTag_names).sum()
    monthTrend = np.isin('trendMonth', ITEMPattern).sum()
    day1effect = np.isin('DAY_1', ITEMPattern).sum()

    if first_peak == 7:
        selected_freq = 7
    elif weekly >= 2:
        selected_freq = 7
    elif daily >= 2 | day1effect == 1:
        selected_freq = train_last_month_day
    elif 25 < first_peak < 34:
        selected_freq = 28
    else:
        selected_freq = 0

    if selected_freq != 0:
        kwargs = {'m': selected_freq, 'with_intercept': False}  # 'seasonal_test': 'ch',

        min_pvalue_sig_week = min(pvalue_sig_week) if pvalue_sig_week else 1
        min_pvalue_sig_month = min(pvalue_sig_month) if pvalue_sig_month else 1

        if (min_pvalue_sig_week < min_pvalue_sig_month < 0.05):
            selected_freq = 7
            kwargs = {'m': selected_freq}  # 'seasonal_test': 'ch',

        if monthTrend > 0:
            kwargs.update({'with_intercept': True})

        if (df.tail(30).View.min() <= df.View.min() or
            df.tail(30).View.max() >= df.View.max()) and (
                kwargs['m'] >= 28 or kwargs['m'] == 7):
            kwargs.update({'with_intercept': False})

        use_frequency = True

    else:
        use_frequency = False
        kwargs = {'with_intercept': True}
        # specified as a string where ‘c’ indicates a constant
        # non-seasonal and has intercept

    return use_frequency, kwargs