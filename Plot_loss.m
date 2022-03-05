load('Processed_data/train_loss_8layers.txt');
load('Processed_data/val_loss_8layers.txt');

figure
hold on
plot(train_loss_8layers,'b','linewidth',1)
plot(val_loss_8layers,'r','linewidth',1)
ylabel('Loss')
xlabel('Epoch')
legend('training loss','validation loss')
set(gcf,'position',[360   556   200   160])