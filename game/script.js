var config = {
  position: 'start',
  draggable: true
}
// http://sam-koblenski.blogspot.com/2017/06/a-barely-adequate-guide-to-displaying.html
var board = ChessBoard('board', config);
var game = new Chess();

 // 1. Load a PGN into the game
 // 2. Get the full move history
 // 3. If Next button clicked, move forward one
 // 4. If Prev button clicked, move backward one
 // 5. If Start button clicked, go to start position
 // 6. If End button clicked, go to end position

var pgn = '1.e4 e5 2.Nf3 Nf6 3.Nc3 d5 4.exd5 Nxd5 5.Bc4 Nf4 6.O-O e4 7.Re1 Kd7 8.Rxe4 Qg5 9.Nxg5 f6 10.Qg4+ Ne6 11.Qxe6+ Kd8 12.Qe8#  1-0';
game.load_pgn(pgn);
$('#pgn5').html(pgn);

var history = game.history();
game.reset();
var i = 0;

var historyMoves = pgn.split('.')
console.log(historyMoves)
console.log(history)

$('#nextBtn5').on('click', function() {
    game.move(history[i]);
    board.position(game.fen());
    i += 1;
    if (i > history.length) {
      i = history.length;
    }
  });

  $('#prevBtn5').on('click', function() {
    game.undo();
    board.position(game.fen());
    i -= 1;
    if (i < 0) {
      i = 0;
    }
  });

  $('#startPositionBtn5').on('click', function() {
    game.reset();
    board.start();
    i = 0;
  });

  // 6. If End button clicked, go to end position
  $('#endPositionBtn5').on('click', function() {
    game.load_pgn(pgn);
    board.position(game.fen());
    i = history.length;
  });



//
// /* board visualization and games state handling */
//
// var onDragStart = function (source, piece, position, orientation) {
//   print("In onDragStart")
//     if (game.in_checkmate() === true || game.in_draw() === true ||
//         piece.search(/^b/) !== -1) {
//         print("End condition")
//         return false;
//     }
// };
//
// var makeBestMove = function () {
//     var bestMove = getBestMove(game);
//     game.ugly_move(bestMove);
//     board.position(game.fen());
//     renderMoveHistory(game.history());
//     if (game.game_over()) {
//         alert('Game over');
//     }
// };
//
//
// var positionCount;
// var getBestMove = function (game) {
//     if (game.game_over()) {
//         alert('Game over');
//     }
//
//     positionCount = 0;
//     var depth = parseInt($('#search-depth').find(':selected').text());
//
//     var d = new Date().getTime();
//     var bestMove = minimaxRoot(depth, game, true);
//     var d2 = new Date().getTime();
//     var moveTime = (d2 - d);
//     var positionsPerS = ( positionCount * 1000 / moveTime);
//
//     $('#position-count').text(positionCount);
//     $('#time').text(moveTime/1000 + 's');
//     $('#positions-per-s').text(positionsPerS);
//     return bestMove;
// };
//
// var renderMoveHistory = function (moves) {
//     var historyElement = $('#move-history').empty();
//     historyElement.empty();
//     for (var i = 0; i < moves.length; i = i + 2) {
//         historyElement.append('<span>' + moves[i] + ' ' + ( moves[i + 1] ? moves[i + 1] : ' ') + '</span><br>')
//     }
//     historyElement.scrollTop(historyElement[0].scrollHeight);
//
// };
//
// var onDrop = function (source, target) {
//
//     var move = game.move({
//         from: source,
//         to: target,
//         promotion: 'q'
//     });
//
//     removeGreySquares();
//     if (move === null) {
//         return 'snapback';
//     }
//
//     renderMoveHistory(game.history());
//     window.setTimeout(makeBestMove, 250);
// };
//
// var onSnapEnd = function () {
//     board.position(game.fen());
// };
//
// var onMouseoverSquare = function(square, piece) {
//     var moves = game.moves({
//         square: square,
//         verbose: true
//     });
//
//     if (moves.length === 0) return;
//
//     greySquare(square);
//
//     for (var i = 0; i < moves.length; i++) {
//         greySquare(moves[i].to);
//     }
// };
//
// var onMouseoutSquare = function(square, piece) {
//     removeGreySquares();
// };
//
// var removeGreySquares = function() {
//     $('#board .square-55d63').css('background', '');
// };
//
// var greySquare = function(square) {
//     var squareEl = $('#board .square-' + square);
//
//     var background = '#a9a9a9';
//     if (squareEl.hasClass('black-3c85d') === true) {
//         background = '#696969';
//     }
//
//     squareEl.css('background', background);
// };
//
// var config = {
//   position: 'start',
//   draggable: true,
//   onDragStart: onDragStart,
//   onDrop: onDrop,
//   onMouseoutSquare: onMouseoutSquare,
//   onMouseoverSquare: onMouseoverSquare,
//   onSnapEnd: onSnapEnd
// }
//
//
// var board = ChessBoard('board', config);
