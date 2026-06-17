from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='reptile city history weather')
parser.add_argument('--area', '-a', type=str, default='chongqing', help='defalt:chongqing')
parser.add_argument('--save-path', '-s', type=str, default='./reptile/output', help='save path，default:./reptile/output')
parser.add_argument('--slow',action='store_true',default=False, help='slow mode')
parser.add_argument('--begin-year', '-b', type=int, default=2011, help='begin year')
parser.add_argument('--end-year', '-e', type=int, default=2025, help='end year')
args = parser.parse_args()

area = args.area
save_path = args.save_path
mode=args.slow
begin_year = args.begin_year
end_year = args.end_year

os.makedirs(save_path, exist_ok=True)

driver = webdriver.Edge()

data=[]
for i in range(begin_year,end_year+1):
    for j in range(1,13):
        if(j>=10):
            url='https://lishi.tianqi.com/'+area+'/'+str(i)+str(j)+'.html'
        else:
            url='https://lishi.tianqi.com/'+area+'/'+str(i)+'0'+str(j)+'.html'
        driver.get(url)
        button=driver.find_element(By.CLASS_NAME,"lishidesc2")
        button.click()
        if mode:
            time.sleep(5)
        else:
            time.sleep(1)
        list_tag=driver.find_element(By.CLASS_NAME,"tian_three")
        items=list_tag.find_elements(By.TAG_NAME,"li")
        for item in items:
            day=item.find_element(By.CLASS_NAME,"th200")
            massage=item.find_elements(By.CLASS_NAME,"th140")
            maxw=massage[0]
            minw=massage[1]
            whther=massage[2]
            wind=massage[3]
            whether={
                '日期':day.text,
                '最高温度':maxw.text,
                '最低温度':minw.text,
                '天气':whther.text,
                '风向':wind.text,
            }
            print(whether)
            data.append(whether)
        time.sleep(2)

os.makedirs(save_path, exist_ok=True)
df = pd.DataFrame(data)
output_file = os.path.join(save_path, 'Whether.xlsx')
df.to_excel(output_file, index=False)
print(f'Initial datas is save as {output_file}')
print('\nBegin to wash data')
data = pd.read_excel(output_file)
data['日期'] = data['日期'].str.replace(' 周一', '')
data['日期'] = data['日期'].str.replace(' 周二', '')
data['日期'] = data['日期'].str.replace(' 周三', '')
data['日期'] = data['日期'].str.replace(' 周四', '')
data['日期'] = data['日期'].str.replace(' 周五', '')
data['日期'] = data['日期'].str.replace(' 周六', '')
data['日期'] = data['日期'].str.replace(' 周日', '')
data['日期'] = data['日期'].str.replace(' 星期一', '')
data['日期'] = data['日期'].str.replace(' 星期二', '')
data['日期'] = data['日期'].str.replace(' 星期三', '')
data['日期'] = data['日期'].str.replace(' 星期四', '')
data['日期'] = data['日期'].str.replace(' 星期五', '')
data['日期'] = data['日期'].str.replace(' 星期六', '')
data['日期'] = data['日期'].str.replace(' 星期日', '')
data['日期'] = data['日期'].str.replace(' null', '')
data['最高温度'] = data['最高温度'].str.replace('℃', '')
data['最低温度'] = data['最低温度'].str.replace('℃', '')
data.to_excel(output_file, index=False)
print('\nData washed successfully')
print('\nBegin to make charts')
df = pd.read_excel(output_file)
df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')
df['年份'] = df['日期'].dt.year
df['月份'] = df['日期'].dt.strftime('%m-%d')
pivot_table = df.pivot(index='年份', columns='月份', values='最高温度')
pivot_path = os.path.join(save_path, 'pivot/MaxTemperature.xlsx')
pivot_dir = os.path.dirname(pivot_path)
os.makedirs(pivot_dir, exist_ok=True)
pivot_table.to_excel(pivot_path)
print(f'MaxTemperature Pivot table is save as {pivot_path}')
df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')
df['年份'] = df['日期'].dt.year
df['月份'] = df['日期'].dt.strftime('%m-%d')
pivot_table = df.pivot(index='年份', columns='月份', values='最低温度')
pivot_path = os.path.join(save_path, 'pivot/MinTemperature.xlsx')
pivot_table.to_excel(pivot_path)
print(f'MinTemperature Pivot table is save as {pivot_path}')