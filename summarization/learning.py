import torch
from torch import autograd, nn, optim
from random import shuffle
import glob
import pickle
import math
import random

from Decoder import Decoder
from Encoder import Encoder
from Vocab import Vocab, SOS_TOKEN, EOS_TOKEN


def load_data():
    chunks = glob.glob("./chunked_data/*.pkl")
    shuffle(chunks)
    print(f"found {len(chunks)} chunks")

    train_num = math.floor(0.85 * len(chunks))

    return chunks[:train_num], chunks[train_num:]


def create_vocab(data_chunk):
    vocab = Vocab()

    for chunk in data_chunk:
        with open(chunk, 'rb') as data_chunk:
            data = pickle.load(data_chunk)
            for record in data:
                for sent in record["story"]:
                    vocab.index_sentence(sent)
    return vocab


train_data, eval_data = load_data()

hidden_size = 500
embedding_size = 100
n_layers = 2
dropout_p = 0.05

max_length = 60

vocab = create_vocab(train_data)

print(f"found {vocab.n_words} distinct words ")

# Initialize models
encoder = Encoder(vocab.n_words, embedding_size, hidden_size, n_layers).cuda()
decoder = Decoder(hidden_size, embedding_size, vocab.n_words, n_layers).cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

n_epochs = 1000
print_every = 200

print_loss_total = 0 # Reset every print_every


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Get size of input and target sentences
    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_TOKEN]]).cuda()
    decoder_context = torch.zeros(1, decoder.hidden_size).cuda()
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden = decoder(decoder_input, decoder_context, decoder_hidden,
                                                                      encoder_outputs)
            loss += criterion(decoder_output[0].view(-1).unsqueeze(0), target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di]  # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden = decoder(decoder_input, decoder_context, decoder_hidden,
                                                                      encoder_outputs)
            loss += criterion(decoder_output[0].view(-1).unsqueeze(0), target_tensor[di].unsqueeze(0))

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.topk(1)
            ni = topi[0][0]

            decoder_input = torch.LongTensor([[ni]]).cuda()  # Chosen word is next input

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_TOKEN:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


for epoch in range(1, n_epochs):
    count = 0
    for train_chunk in train_data:
        with open(train_chunk, 'rb') as train_data:
            data_chunk = pickle.load(train_data)
            for record in data_chunk:
                count +=1
                print(count)
                abstract = ''.join(record["highlights"])
                story = ''.join(record["story"])

                if count % 200 == 0:
                    encoder.save('./models/encoser.pt')
                    decoder.save('./models/decoder.pt')

                input_tensor = torch.tensor(vocab.index_sentence(story, write=False)).cuda()
                target_tensor = torch.tensor(vocab.index_sentence(abstract, write=False)).cuda()

                # Run the train function
                loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

                # Keep track of loss
                print_loss_total += loss

                if epoch == 0: continue

                if epoch % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('(Epoch %d) %.4f' % (epoch, print_loss_avg))


def evaluate(story, highlight_max_length):
    input_tensor = torch.tensor(vocab.index_sentence(story, False)).cuda()

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[SOS_TOKEN]]).cuda()  # SOS
    decoder_context = torch.zeros(1, decoder.hidden_size).cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []

    # Run through decoder
    for di in range(highlight_max_length):
        decoder_output, decoder_context, decoder_hidden = decoder(decoder_input, decoder_context, decoder_hidden,
                                                                  encoder_outputs)
        # Choose top word from output
        topv, topi = decoder_output.cpu().topk(1)
        ni = topi[0][0].item()
        if ni == SOS_TOKEN:
            continue
        if ni == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocab.index2word[ni])

        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]]).cuda()

    return decoded_words


def evaluate_randomly(eval_data):
    eval_chunk = random.choice(eval_data)
    count = 0
    with open(eval_chunk, 'rb') as eval_data:
        data_chunk = pickle.load(eval_data)
        for record in data_chunk:
            count += 1
            abstract = ''.join(record["highlights"])
            story = ''.join(record["story"])
            output_words = evaluate(story, max_length)
            output_sentence = ' '.join(output_words)

            print(f"Original: {abstract} \n\n")
            print(f"Prediction: {output_sentence}")


encoder.save('./models/encoser.pt')
decoder.save('./models/decoder.pt')

evaluate_randomly()
