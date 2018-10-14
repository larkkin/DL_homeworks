import time

__all__ = ['ResNeXt', 'resnext']

class Logger:
    def __init__(self, path, n_batches):
        self.lines = []
        self.path = path
        self.training_start_time = time.time()
        self.losses = []
        self.n_batches = n_batches
    def log(self, start_time, iteration, epoch, running_loss):
        print_every = self.n_batches // 10
        if (iteration + 1) % (print_every + 1) == 0:
            self.lines.append("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1,
                int(100 * (iteration+1) / self.n_batches), running_loss / print_every, time.time() - start_time))
            #Reset running loss and time
            self.losses.append(running_loss / print_every)
            running_loss = 0.0
            start_time = time.time()
        return start_time, running_loss
    def end(self):
        self.lines.append("Training finished, took {:.2f}s".format(time.time() - self.training_start_time))
        with open(self.path, 'w') as otp:
            otp.write('\n'.join(self.lines))
