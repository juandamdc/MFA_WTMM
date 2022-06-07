
def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    print(f'Time Lapsed = {int(hours)}:{int(mins)}:{sec}')