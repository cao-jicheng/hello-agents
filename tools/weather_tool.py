import requests

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        nearest_area = data["nearest_area"][0]["areaName"][0]["value"]
        current_condition = data["current_condition"][0]
        humidity = current_condition["humidity"]
        visibility = current_condition["visibility"]
        temp_c = current_condition["temp_C"]
        weather = current_condition["weatherDesc"][0]["value"]
        return f"城市{nearest_area}，当前天气{weather}， 温度{temp_c}摄氏度，湿度{humidity}%，能见度{visibility}级"
    except requests.exceptions.RequestException as e:
        return f"⛔ 查询天气时遇到网络问题：{e}"
    except (KeyError, IndexError) as e:
        return f"⛔ 解析天气数据失败：{e}"



