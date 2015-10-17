from scripts.trainer import Trainer
from scripts.tester import Tester

training_dir = 'data/training/'
test_dir = 'test/'
result_dir = 'result/'

# training
trainer = Trainer(training_dir)

# estimated parameters after training
mean_0 = trainer.mean_0
mean_1 = trainer.mean_1
var_0 = trainer.var_0
var_1 = trainer.var_1
bern_lamda = trainer.bern_lamda

# test
tester = Tester(test_dir, result_dir)
tester.begin_test(mean_0, mean_1, var_0, var_1, bern_lamda)
