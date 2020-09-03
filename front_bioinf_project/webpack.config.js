const path = require("path");

let srcPath = path.join(__dirname, "src");
module.exports = {
    devServer: {
        contentBase: srcPath,
        compress: true,
        host: "127.0.0.1",
        port: 8080
    }
};