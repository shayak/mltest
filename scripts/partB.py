import datetime
import myUtilities

with open('dates.txt', 'r') as infile, open('ages.txt', 'w') as outfile:
    for datestr in infile:
        age = myUtilities.age(
            datetime.datetime.strptime(datestr.strip(), '%Y-%m-%d').date()
        )
        outfile.write(f'The age based on {datestr.strip()} is {age} years\n')
