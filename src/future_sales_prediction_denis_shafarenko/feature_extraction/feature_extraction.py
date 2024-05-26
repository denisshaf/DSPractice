import os, re, gc
import itertools
from typing import TypedDict 

import pandas as pd
import numpy as np

from ..utils.type_conversion import squeeze_float, squeeze_int


class LagConfig(TypedDict):
    groupby: list[str]
    column: str
    lags: list[int]


def read_data(input_dir: str) -> dict[str, pd.DataFrame]:
    data = {}

    for file in os.listdir(input_dir):
        data[file.split('.')[0]] = pd.read_csv(f'{input_dir}/{file}')

    return data


replace_shop_ids = {57: 0, 58: 1, 11: 10}


def transform_shops(data: dict[str, pd.DataFrame]) -> pd.DataFrame:

    shop_types = ['ТЦ', 'ТРК', 'ТРЦ', 'МТРЦ', 'Магазин', 'ТК']
    exceptions = ['Выездная Торговля', 'Интернет-магазин ЧС', 'Цифровой склад 1С-Онлайн']
    exceptions_ind = list(data['shops']['shop_name'][data['shops']['shop_name'].isin(exceptions)].index)
    online_shops = ['Интернет-магазин ЧС', 'Цифровой склад 1С-Онлайн']
    online_shops_ind = list(data['shops']['shop_name'][data['shops']['shop_name'].isin(online_shops)].index)

    regexpr = re.compile(r'[^"\W][\w-]+|".+?"')
    tokens = [regexpr.findall(sent) for sent in data['shops']['shop_name']]

    def parse_shops(sentences):
        shops: dict = {'city': [None] * len(sentences), 'type': [None] * len(sentences)}

        for i, tokens in enumerate(sentences):
            if i not in exceptions_ind:
                shops['city'][i] = tokens[0]

                if tokens[1] in shop_types:
                    shops['type'][i] = tokens[1]
                else:
                    shops['type'][i] = 'Магазин'

        return shops

    shops = pd.DataFrame.from_dict(parse_shops(tokens))

    shops.loc[online_shops_ind, 'city'] = 'Интернет'
    shops.loc[online_shops_ind, 'type'] = 'Интернет-магазин'
    shops.fillna({'city': ''}, inplace=True)
    shops.fillna({'type': 'Магазин'}, inplace=True)
    shops.replace('МТРЦ', 'ТРЦ', inplace=True)

    return shops


def transform_item_categories(data: dict[str, pd.DataFrame]) -> pd.DataFrame:

    item_categories = data['item_categories'].copy()

    item_categories.replace('Чистые носители (шпиль)', 'Чистые носители - (шпиль)', inplace=True)
    item_categories.replace('Чистые носители (штучные)', 'Чистые носители - (штучные)', inplace=True)

    item_categories['split'] = item_categories['item_category_name'].apply(lambda s: [token.strip() for token in s.split(' - ')])
    item_categories['category'] = item_categories['split'].apply(lambda l: l[0])
    item_categories['subcategory'] = item_categories['split'].apply(lambda l: l[1] if len(l) > 1 else '')
    item_categories['subcategory'] = np.where(item_categories['category'].isin(['Игры Android', 'Игры MAC']),
                                              item_categories['category'].apply(lambda s: s.split()[-1]),
                                              item_categories['subcategory'])
    item_categories['category'] = np.where(item_categories['category'].isin(['Игры Android', 'Игры MAC']),
                                              'Игры',
                                              item_categories['category'])
    
    rare_categories = item_categories['category'].value_counts()[item_categories['category'].value_counts() < 2].index
    item_categories.replace(dict.fromkeys(rare_categories, 'Другое'), inplace=True)

    item_categories.drop(columns=['split', 'item_category_name'], inplace=True, errors='ignore')

    return item_categories


def transform_items(data: dict[str, pd.DataFrame]) -> pd.DataFrame:

    items = data['items'].copy()

    def name_correction(x):
        x = x.lower()
        x = x.partition('[')[0]
        x = x.partition('(')[0]
        x = re.sub('[^A-Za-z0-9А-Яа-яё]+', ' ', x)
        x = x.replace('  ', ' ')
        x = x.replace('ё', 'е')
        x = x.strip()
        return x

    # split item names by first bracket
    items["name2"] = items.item_name.str.split("[", n=1).str[1]
    items["name3"] = items.item_name.str.split("(", n=1).str[1]

    # replace special characters and turn to lower case
    items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-яё]+', " ").str.lower()
    items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-яё]+', " ").str.lower()

    # fill nulls with '0'
    items = items.fillna('0')

    items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

    # return all characters except the last if name 2 is not "0" - the closing bracket
    items['name2'] = items['name2'].apply(lambda x: x[:-1] if x != "0" else "0")

    items["type"] = items['name2'].apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0])
    items.loc[(items['type'] == "x360") | (items['type'] == "xbox360") | (items['type'] == "xbox 360") , "type"] = "xbox 360"
    items.loc[(items['type'] == "pc") | (items['type'] == "рс") | (items['type'] == "pс") | (items['type'] == "рc"), "type"] = "pc"
    items.loc[items['type'] == 'рs3' , "type"] = "ps3"
    items['type'] = items['type'].apply(lambda x: x.replace(" ", ""))

    group_sum = items.groupby(["type"]).agg({"item_id": "count"})
    group_sum = group_sum.reset_index()
    drop_cols = []
    for cat in group_sum['type'].unique():
        if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] < 40:
            drop_cols.append(cat)
    items['name2'] = items['name2'].apply( lambda x: "other" if (x in drop_cols) else x )
    items = items.drop(columns=["type"])

    items.drop(columns=["item_name"], inplace=True, errors='ignore')

    rare_values_name2 = items['name2'].value_counts()[items['name2'].value_counts() < 2].keys()
    rare_values_name3 = items['name3'].value_counts()[items['name3'].value_counts() < 2].keys()

    items['name2'] = items['name2'].replace(rare_values_name2, 'other')
    items['name3'] = items['name3'].replace(rare_values_name3, 'other')

    return items


def transform_numeric_features(data: dict[str, pd.DataFrame], items: pd.DataFrame, shops: pd.DataFrame) -> pd.DataFrame:

    month_sales = data['sales_train'][['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]
    month_sales['shop_id'] = month_sales['shop_id'].replace(replace_shop_ids)
    month_sales = month_sales.groupby(by=['date_block_num', 'shop_id', 'item_id']) \
                            .agg(avg_price=('item_price', 'mean'), item_cnt_month=('item_cnt_day', 'sum'))
    month_sales = month_sales.reset_index(level=[1, 2])

    month_sales['shop_id'] = squeeze_int(month_sales['shop_id'])
    month_sales['item_id'] = squeeze_int(month_sales['item_id'])
    month_sales['avg_price'] = squeeze_float(month_sales['avg_price'])
    month_sales['item_cnt_month'] = squeeze_float(month_sales['item_cnt_month'])

    shop_item_combinations = pd.DataFrame(
        list(itertools.product(shops.index.delete(list(replace_shop_ids.keys())), items['item_id'])), \
        columns=['shop_id', 'item_id']
    )
    shop_item_combinations['shop_id'] = squeeze_int(shop_item_combinations['shop_id'])
    shop_item_combinations['item_id'] = squeeze_int(shop_item_combinations['item_id'])

    month_frames = [
        month_sales.loc[i].reset_index() \
                    .rename(columns={'index': 'date_block_num'}) \
                    .merge(shop_item_combinations, how='outer', on=['shop_id', 'item_id']) 
        for i in range(month_sales.index.max() + 1)
    ]

    for i in range(len(month_frames)):
        month_frames[i]['date_block_num'] = squeeze_float(month_frames[i]['date_block_num'])
        month_frames[i]['date_block_num'].fillna(i, inplace=True)
        month_frames[i]['avg_price'].fillna(0, inplace=True)
        month_frames[i]['item_cnt_month'].fillna(0, inplace=True)

    month_data = pd.concat(month_frames, ignore_index=True)
    
    month_data['item_revenue'] = month_data['avg_price'].astype(np.float64) * month_data['item_cnt_month'].astype(np.float64)
    month_data['shop_month_revenue'] = month_data[['date_block_num', 'shop_id', 'item_revenue']].astype(np.float64).groupby(['date_block_num', 'shop_id'], as_index=False)['item_revenue'].transform('sum')
    month_data['shop_month_revenue'] = squeeze_float(month_data['shop_month_revenue'])

    month_data['avg_item_revenue'] = month_data.groupby(['shop_id', 'item_id']).transform('mean')['item_revenue']
    month_data['relative_delta_revenue'] = (month_data['item_revenue'] - month_data['avg_item_revenue']) / month_data['avg_item_revenue']

    month_data['relative_delta_revenue'].fillna(0, inplace=True)
    month_data.replace([np.inf], 0, inplace=True)

    month_data['item_revenue'] = squeeze_float(month_data['item_revenue'])
    month_data['avg_item_revenue'] = squeeze_float(month_data['avg_item_revenue'])
    month_data['relative_delta_revenue'] = squeeze_float(month_data['relative_delta_revenue'])

    month_data['avg_shop_month_revenue'] = month_data.groupby(['shop_id', 'item_id']) \
                                                    .transform('mean')['shop_month_revenue']
    month_data['relative_shop_delta_revenue'] = (month_data['shop_month_revenue'] - month_data['avg_shop_month_revenue']) / month_data['avg_shop_month_revenue']

    month_data['relative_shop_delta_revenue'] = squeeze_float(month_data['relative_shop_delta_revenue'])

    month_data.drop(columns=['avg_item_revenue', 'avg_shop_month_revenue'], inplace=True)

    return month_data


def create_lags(numeric_features: pd.DataFrame, lag_config_list: list[LagConfig]) -> None:
    """ Creates lag features inplace"""

    def create_lags(df, groupby, column, lags):
        group = df[groupby + [column]].astype(np.float32).groupby(groupby).mean().groupby(level=0)
        for lag in lags:
            m = group.shift(lag).rename(columns={column: f'{column}_lag_{lag}'}).reset_index()
            numeric_features[f'{column}_lag_{lag}'] = numeric_features.merge(m, on=groupby)[f'{column}_lag_{lag}']
            gc.collect()


    for lag_config in lag_config_list:
        create_lags(numeric_features, **lag_config)
        
        for lag in lag_config['lags']:
            column_name = f'{lag_config["column"]}_lag_{lag}'
            numeric_features[column_name] = squeeze_float(numeric_features[column_name])


def create_date_features(numeric_features: pd.DataFrame) -> None:
    """Creates date features inplace"""

    numeric_features['month'] = squeeze_int(numeric_features['date_block_num'] % 12)
    numeric_features['year'] = squeeze_int(numeric_features['date_block_num'] // 12)

    numeric_features['month_x'] = (numeric_features['month'].astype('float32') * np.pi / 6).apply(np.sin)
    numeric_features['month_y'] = (numeric_features['month'].astype('float32') * np.pi / 6).apply(np.cos)


def run_pipeline(input_dir: str, output_dir: str, config: dict) -> pd.DataFrame:
    """ Runs data transformation pipline with filling Nones with zeros

    Arguments:
    input_dir - path to directory with raw data
    output_dir - path to directory for storing transformed data data
    """

    data = read_data(input_dir)
    shops = transform_shops(data)
    item_categories = transform_item_categories(data)
    items = transform_items(data)
    numeric_features = transform_numeric_features(data, items, shops)
    
    gc.collect()

    lags_for_prediction_columns: list[LagConfig] = config['lags']
    create_lags(numeric_features, lags_for_prediction_columns)

    # drop redundant features and save them only from the last month to create lags for submission
    lag_columns_to_drop = [lag['column'] for lag in lags_for_prediction_columns].remove(config['target'])
    numeric_features.drop(columns=lag_columns_to_drop, inplace=True)

    last_month_num = numeric_features['date_block_num'][-1]
    last_month_lags = numeric_features.query(f'date_block_num == {last_month_num}')[lag_columns_to_drop]

    create_date_features(numeric_features)

    features = numeric_features.merge(shops, how='left', left_on='shop_id', right_index=True) \
                           .merge(items, how='left', on='item_id') \
                           .merge(item_categories, how='left', on='item_category_id')


    # save features

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, '')   # add trailing slash if not present

    features.to_parquet(f'{output_dir}features.parquet', index=True)
    last_month_lags.to_parquet(f'{output_dir}last_month_lags.parquet', index=True)

    return features