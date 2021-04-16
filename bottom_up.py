import re
import pandas as pd


class BottomUp:
    def __init__(self, table_labels_file='labels/label_space_with_province.tsv'):
        """
        table_labels_file: str = file name that stores hierarchical labels, which will be used to fill up at bottom up
        """
        self.labels = pd.read_csv(table_labels_file, delimiter='\t', header=0)
        self.cities = self.labels['dialect_city_id'].unique().tolist()
        self.provinces = self.labels['dialect_province_id'].unique().tolist()
        self.countries = self.labels['dialect_country_id'].unique(
        ).tolist()
        self.regions = self.labels['dialect_region_id'].unique(
        ).tolist()

        self.gold_city = dict()
        self.gold_country = dict()
        self.gold_region = dict()
        for city in self.cities:
            self.gold_city[city] = 0.0

        for country in self.countries:
            self.gold_country[country] = 0.0

        for region in self.regions:
            self.gold_region[region] = 0.0

    def standardize_df(self, df):
        for index in df.index:
            if type(df.loc[index, 'dialect_city_id']) == str and len(df.loc[index, 'dialect_city_id']) > 0 and df.loc[index, 'dialect_city_id'] != 'NaN':
                df.loc[index] = self.standardize_row(df.loc[index], 'city')
            elif type(df.loc[index, 'dialect_province_id']) == str and len(df.loc[index, 'dialect_province_id']) > 0 and df.loc[index, 'dialect_province_id'] != 'NaN':
                df.loc[index] = self.standardize_row(df.loc[index], 'province')
            elif type(df.loc[index, 'dialect_country_id']) == str and len(df.loc[index, 'dialect_country_id']) > 0 and df.loc[index, 'dialect_country_id'] != 'NaN':
                df.loc[index] = self.standardize_row(df.loc[index], 'country')
            elif type(df.loc[index, 'dialect_region_id']) == str and len(df.loc[index, 'dialect_region_id']) > 0 and df.loc[index, 'dialect_region_id'] != 'NaN':
                df.loc[index] = self.standardize_row(df.loc[index], 'region')

        return df

    def preprocess_label(self, labels_row, level):
        return re.sub('[^a-z]+', '_', labels_row[f'dialect_{level}_id'].lower().strip())

    def get_gold(self, row, level, label):
        gold_city = self.gold_city
        gold_country = self.gold_country
        gold_region = self.gold_region

        if not level:
            for city in self.cities:
                gold_city[city] = 1.0/float(len(self.cities))

            for country in self.countries:
                gold_country[country] = 1.0/float(len(self.countries))

            for region in self.regions:
                gold_region[region] = 1.0/float(len(self.regions))

            return gold_city, gold_country, gold_region

        if level == 'region':
            gold_region[label] = 1.0
            # distribute probability equally to lower level
            countries_under_region = self.labels.loc[self.labels['dialect_region_id']
                                                     == label].dialect_country_id.unique().tolist()
            cities_under_region = []
            for country in countries_under_region:
                gold_country[country] = 1.0/float(len(countries_under_region))
                cities_under_region += [i[0]
                                        for i in self.labels.loc[self.labels['dialect_country_id'] == country].values]
            for city in cities_under_region:
                gold_city[city] = 1.0/float(len(cities_under_region))

        elif level == 'country':
            gold_country[label] = 1.0
            cities_under_region = self.labels.loc[self.labels['dialect_country_id'] == label].values.unique(
            )
            gold_region[cities_under_region[0][-1]] = 1.0
            for city in cities_under_region:
                gold_city[city[0]] = 1.0/float(len(cities_under_region))

        elif level == 'city':
            gold_city[label] = 1.0
            labels = self.labels.loc[self.labels['dialect_city_id'] ==
                                     label].values[0]
            gold_region[labels[-1]] = 1.0
            gold_country[labels[-2]] = 1.0

        return gold_city, gold_country, gold_region

    def standardize_row(self, labels_row, level):

        label = self.preprocess_label(labels_row, level)

        if label in self.labels[[f'dialect_{level}_id']].values:
            row = self.labels.loc[self.labels[f'dialect_{level}_id']
                                  == label]

            if len(row) > 1:
                try:
                    row_new = row.loc[row['dialect_region_id'] ==
                                      labels_row['dialect_region_id']].iloc[:1]
                    # iraq and basra is the case when region is nothing because changed to gulf
                    if len(row_new) == 0:
                        row_new = row.loc[row['dialect_country_id'] ==
                                          labels_row['dialect_country_id']].iloc[:1]
                    if len(row_new) == 0:
                        row_new = row.iloc[:1]
                    row = row_new
                except:
                    print(f'error  {row.values}')

            if level == 'city' or level == 'province':
                labels_row['dialect_city_id'] = row['dialect_city_id'].values[0]
                labels_row['dialect_province_id'] = row['dialect_province_id'].values[0]
            if level == 'city' or level == 'province' or level == 'country':
                labels_row['dialect_country_id'] = row['dialect_country_id'].values[0]
            labels_row['dialect_region_id'] = row['dialect_region_id'].values[0]

        return labels_row
