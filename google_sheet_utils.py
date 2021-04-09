import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


def get_sheet(sheet_name, credentials_json='google_credentials.json'):

    # define the scope
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        credentials_json, scope)

    # authorize the clientsheet
    client = gspread.authorize(creds)

    # get the instance of the Spreadsheet
    sheet = client.open(sheet_name)

    return sheet


def get_all_records(sheet, page):
    """
    Returns all data from sheet
     Args:
      sheet: google sheet
      page: index of the page in the sheet
    Returns:
      data: Pandas dataframe containing all data
    """

    sheet_instance = sheet.get_worksheet(page)
    records_dict = sheet_instance.get_all_records()
    return pd.DataFrame.from_dict(records_dict)


def rewrite_page(sheet, page, df):
    """
    Rewrites page with pandas DataFrame
     Args:
      sheet: google sheet
      page: index of the page in the sheet
      df: Pandas dataframe to update
    """
    sheet_instance = sheet.get_worksheet(page)
    sheet_instance.update([df.columns.values.tolist()] +
                          df.values.tolist())


def create_page(sheet, title, rows, cols):
    """
    Creates page in the sheet
     Args:
      sheet: google sheet
      title: title for the new page
      rows: string indicating number  of rows
      cols: string indicating number  of cols
    """
    sheet.add_worksheet(title=title, rows=rows, cols=cols)


def find_idx_page(sheet, title):
    """
    Finds index of the page given the title of the page
    Args:
      sheet: google sheet
      title: title for the new page
    Returns:
        index of page, -1 if not found
    """
    sheet_list = sheet.worksheets()
    for i in range(len(sheet_list)):
        if sheet_list[i].title == title:
            return i

    return -1
