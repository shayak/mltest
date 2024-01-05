import datetime


def getName():
    name = input('Enter name: ')
    return name


def getBirthdate():
    bday = input('Enter birthday (yyyy-mm-dd): ')
    birthdate = datetime.datetime.strptime(bday, '%Y-%m-%d').date()
    return birthdate


def age(input_date):
    age_in_days = (datetime.date.today() - input_date).days
    age = age_in_days // 365
    return age
