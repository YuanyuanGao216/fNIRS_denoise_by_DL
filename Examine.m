clear all
close all

load('Processed_data/random_profile.mat','p')
load('Processed_data/HRF_test.mat','HRF_test')
load('Processed_data/HRF_test_noised.mat','HRF_test_noised')
load('Processed_data/Noise_test.mat','Noise_test')

index = 1000;
figure
hold on;
plot(HRF_test(index,:),'b')
plot(HRF_test_noised(index,:),'r')
plot(Noise_test(index,:),'k')

