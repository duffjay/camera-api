from datetime import datetime

date_str = input('Enter date (May 12 2020 12:40PM):')
date_obj = datetime.strptime(date_str, '%b %d %Y %I:%M%p')
print (date_obj.timestamp())

