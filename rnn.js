require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs');
const fs = require('fs')
const path = require('path')
var filePath = path.join(__dirname, 'deathgrips.txt')

max_fatures = 2000
embed_dim = 128
lstm_out = 196

fs.readFile(filePath, {encoding: 'utf-8'}, async function(err,data){
    if (!err) {
        var lines = data.split('\n')
        vocabulary = []
        titles = []
        karma = []
        for (let line of lines) {
          if (line) {
            var line_split = line.split(',')
            var title = line_split[0].toLowerCase()
            var title = title.replace(/[^A-Za-z0-9 ]/g, '');
            for (let word of title.split(' ')) {
              if (!vocabulary.includes(word)) {
                vocabulary.push(word)
              }
            }
            titles.push(title)
            karma.push(parseFloat(line_split[1]))
          }
        }
        dict = {}
        count = 0
        for (let word of vocabulary) {
          dict[word] = count
          count++
        }
        var sequences = []
        var max_len = 0
        for(let title of titles) {
          var words = title.split(' ')
          var arr = []
          for (let word of words) {
            arr.push(dict[word])
          }
          sequences.push(arr)
          if (arr.length > max_len) {
            max_len = arr.length
          }
        }
        var X = []
        var Y = []
        var seq_count = 0
        for (let sequence of sequences) {
          var padded_arr = Array(max_len - sequence.length).fill(0).concat(sequence).slice(0,max_len)
          X.push(padded_arr)
          if (typeof karma[seq_count] == 'number') {
            Y.push(karma[seq_count])
          }
          seq_count++
        }
        var max_karma = Math.max(karma)
        X_tensor = tf.tensor2d(X)
        Y_tensor = tf.tensor1d(Y)
        X_tensor.print(true)
        Y_tensor.print(true)

        var model = tf.sequential();
        model.add(tf.layers.embedding({
          inputDim: vocabulary.length,
          outputDim: 8,
          inputLength: max_len
        }))
        model.add(tf.layers.gru({
          units: 16,
          returnSequences: true
        }));
        model.add(tf.layers.gru({
          units: 8,
          returnSequences: true
        }));
        model.add(tf.layers.gru({
          units: 4
        }));
        model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
        model.summary()

        model.compile({loss: 'binaryCrossentropy', optimizer: tf.train.adam(0.003)});

        await model.fit(X_tensor, Y_tensor, {
          epochs: 7,
          batchSize: 32,
          callbacks: {
            onEpochEnd: async (epoch, logs) => {
              console.log(logs.loss + ",")
            }
          }
        })
        const saveResult = await model.save('file:///tmp/my-model-1');

        twt = 'Meetings: Because none of us is as dumb as all of us.'
        var words = twt.split(' ')
        var arr = []
        for (let word of words) {
          var idx = dict[word]
          arr.push((!idx) ? 0 : idx)
        }
        var padded_arr = Array(max_len - arr.length).fill(0).concat(arr).slice(0,max_len)
        var test = tf.tensor2d([padded_arr])
        const output = model.predict(test)
        var output_data = await output.dataSync()
        console.log(output_data)

    } else {
        console.log(err);
    }
});
