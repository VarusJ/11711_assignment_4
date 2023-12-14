import matplotlib.pyplot as plt

# Data
macro_f1_10 = 0.29673125265904204
macro_f1_20 = 0.3075648198376616
micro_f1_10 = 0.42483907610753496
micro_f1_20 = 0.4672539256989659

# Step sizes (suffixes of the data names)
step_sizes = [10, 20]

# Values for Macro F1 and Micro F1
macro_f1_values = [macro_f1_10, macro_f1_20]
micro_f1_values = [micro_f1_10, micro_f1_20]

# Creating the plot
plt.figure(figsize=(8, 6))
plt.plot(step_sizes, macro_f1_values, label='Macro F1', marker='o')
plt.plot(step_sizes, micro_f1_values, label='Micro F1', marker='o')

# Adding title and labels
plt.title('Train set size = 100')
plt.xlabel('Step Size')
plt.ylabel('Macro F1 and Micro F1')
plt.xticks(step_sizes)
plt.ylim(bottom=0)
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()


# New Data
macro_f1_10 = 0.30330667407186906
macro_f1_20 = 0.3284946164234057
macro_f1_30 = 0.3388598640023898
macro_f1_39 = 0.34105910444763615
micro_f1_10 = 0.4465894465894466
micro_f1_20 = 0.5408362641656896
micro_f1_30 = 0.5763157894736842
micro_f1_39 = 0.5843293492695883

# Step sizes and values for Macro F1 and Micro F1
step_sizes = [10, 20, 30, 39]
macro_f1_values = [macro_f1_10, macro_f1_20, macro_f1_30, macro_f1_39]
micro_f1_values = [micro_f1_10, micro_f1_20, micro_f1_30, micro_f1_39]

# Creating the plot
plt.figure(figsize=(8, 6))
plt.plot(step_sizes, macro_f1_values, label='Macro F1', marker='o')
plt.plot(step_sizes, micro_f1_values, label='Micro F1', marker='o')

# Adding title and labels
plt.title('Train set size = 200')
plt.xlabel('Step Size')
plt.ylabel('Macro F1 and Micro F1')
plt.xticks(step_sizes)
plt.legend()

# Setting y-axis to start from zero
plt.ylim(bottom=0)

# Displaying the plot
plt.grid(True)
plt.show()


# New Data for set size 500
macro_f1_values = [0.2890584090206188, 0.32788192210204, 0.3478383004459929, 0.3583122516709761, 
                   0.3654230970538198, 0.3678253073835534, 0.3724001414537173, 0.375031736016182, 
                   0.3766601741991135, 0.37612723293026323]
micro_f1_values = [0.4088912551577266, 0.5467376830892143, 0.6060935799782372, 0.6302929334999307, 
                   0.6415041782729805, 0.6519960502186487, 0.6582850685125018, 0.6602951510892481, 
                   0.6635698198198198, 0.6628169014084507]
step_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 96]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(step_sizes, macro_f1_values, label='Macro F1', marker='o')
plt.plot(step_sizes, micro_f1_values, label='Micro F1', marker='o')

# Adding title and labels
plt.title('Train set size = 500')
plt.xlabel('Step Size')
plt.ylabel('Macro F1 and Micro F1')
plt.xticks(step_sizes)
plt.legend()

# Setting y-axis to start from zero
plt.ylim(bottom=0)

# Displaying the plot
plt.grid(True)
plt.show()

# New Data for set size 1000
macro_f1_values = [0.25774509282820635, 0.3154202444398421, 0.3310222004105702, 0.3579169131670215, 
                   0.34852632730532834, 0.36802285222570846, 0.37959984193394825, 0.37933140052564646, 
                   0.3902918096799977, 0.39740046414721025, 0.4008917833883324, 0.40428098942350793, 
                   0.4045806332016206, 0.4065068649942984, 0.40959927879338026, 0.40852153574206757, 
                   0.4102311141561885, 0.4113266187557394]
micro_f1_values = [0.34792593655805937, 0.5054593099432231, 0.5793044953350296, 0.6323178016726404, 
                   0.6266829865361077, 0.6562262429679184, 0.6699714586149917, 0.6716372489348753, 
                   0.6876624369362483, 0.6979641446368885, 0.7035007610350076, 0.7096180395766221, 
                   0.7114259316291963, 0.7148125384142593, 0.7202472952086554, 0.7183925811437403, 
                   0.7210128145746488, 0.7221107853726277]
step_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

# Creating the plot
plt.figure(figsize=(12, 6))
plt.plot(step_sizes, macro_f1_values, label='Macro F1', marker='o')
plt.plot(step_sizes, micro_f1_values, label='Micro F1', marker='o')

# Adding title and labels
plt.title('Train set size = 1000')
plt.xlabel('Step Size')
plt.ylabel('Macro F1 and Micro F1')
plt.xticks(step_sizes)
plt.legend()

# Setting y-axis to start from zero
plt.ylim(bottom=0)

# Displaying the plot
plt.grid(True)
plt.show()


# New Data for set size "max"
step_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050]
macro_f1_values = [0.3142309305282578, 0.38198690256633705, 0.4015573841560303, 0.4408710637432846, 
                   0.46772377128270154, 0.4725290296634343, 0.4777594577663282, 0.48739758516822523, 
                   0.48292442960607856, 0.48298004949549, 0.4860294117647058, 0.48092975890078915, 
                   0.4878749615111004, 0.48501036719444873, 0.4843027398130674, 0.48421965565167985, 
                   0.48706460775677385, 0.4847724506997137, 0.48511371298524975, 0.48396478280487776, 
                   0.48452853593767164]
micro_f1_values = [0.503778337531486, 0.6736020806241874, 0.7131578947368421, 0.765625, 
                   0.8183659689399057, 0.8266485384092453, 0.8341296928327645, 0.8499660556687033, 
                   0.8426051560379919, 0.8439862542955326, 0.8479890933878664, 0.8410326086956522, 
                   0.8509189925119128, 0.846945778997941, 0.8458927359131024, 0.8450511945392492, 
                   0.8501026694045174, 0.8471555860178204, 0.848526387936943, 0.8462064251537935, 
                   0.8469945355191257]

# Creating the plot
plt.figure(figsize=(12, 6))
plt.plot(step_sizes, macro_f1_values, label='Macro F1', marker='o')
plt.plot(step_sizes, micro_f1_values, label='Micro F1', marker='o')

# Adding title and labels
plt.title('Train set size = max')
plt.xlabel('Step Size')
plt.ylabel('Macro F1 and Micro F1')
plt.xticks(step_sizes, rotation=45)
plt.legend()

# Setting y-axis to start from zero
plt.ylim(bottom=0)

# Displaying the plot
plt.grid(True)
plt.show()

# Correcting the code to generate the plot

# Maximum Macro F1 and Micro F1 scores for each train set size
max_scores = {
    '100': {'macro_f1': 0.3766601741991135, 'micro_f1': 0.6635698198198198},
    '200': {'macro_f1': 0.34105910444763615, 'micro_f1': 0.5843293492695883},
    '500': {'macro_f1': 0.4113266187557394, 'micro_f1': 0.7221107853726277},
    '1000': {'macro_f1': 0.4878749615111004, 'micro_f1': 0.8509189925119128},
    'max': {'macro_f1': 0.48739758516822523, 'micro_f1': 0.8501026694045174}
}

# Preparing data for the plot
train_set_sizes = ['100', '200', '500', '1000', 'max']
macro_f1_max_values = [max_scores[size]['macro_f1'] for size in train_set_sizes]
micro_f1_max_values = [max_scores[size]['micro_f1'] for size in train_set_sizes]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(train_set_sizes, macro_f1_max_values, label='Max Macro F1', marker='o')
plt.plot(train_set_sizes, micro_f1_max_values, label='Max Micro F1', marker='o')

# Adding title and labels
plt.title('ISEAR (self-reported experiences)')
plt.xlabel('Train Set Size')
plt.ylabel('F1 Score')
plt.xticks(train_set_sizes)
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()
